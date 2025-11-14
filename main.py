from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import openai
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import os
import io
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Mock Interview API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Transcription",
        "X-Should-Load-Next",
        "X-Interview-Complete",
        "X-Current-Question-Index",
        "X-Total-Questions",
        "X-Follow-Up-Count",
    ],
)

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("OPENAI_API_KEY not set in environment")

# Store interview sessions
interview_sessions = {}


class InterviewConfig(BaseModel):
    topic: str
    num_questions: int = 10
    experience_years: int = 0


class InterviewSession(BaseModel):
    session_id: str
    topic: str
    experience_years: int
    questions: List[str]
    current_question_index: int = 0
    conversation_history: List[dict] = []
    follow_up_count: int = 0


def generate_questions(topic: str, experience_years: int, num_questions: int) -> List[str]:
    """Generate interview questions based on topic and experience level using GPT-4"""
    
    # Determine experience level
    if experience_years == 0:
        level = "entry-level or fresher"
        difficulty = "basic concepts, definitions, and fundamental understanding"
    elif experience_years <= 2:
        level = "junior (0-2 years)"
        difficulty = "intermediate concepts, practical applications, and problem-solving"
    elif experience_years <= 5:
        level = "mid-level (3-5 years)"
        difficulty = "advanced concepts, architecture decisions, optimization, and best practices"
    else:
        level = "senior (5+ years)"
        difficulty = "expert-level concepts, system design, performance optimization, team leadership, and architectural patterns"
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    system_prompt = f"""You are an experienced technical interviewer creating {num_questions} interview questions.

Topic: {topic}
Experience Level: {level} ({experience_years} years)
Difficulty Focus: {difficulty}

Generate exactly {num_questions} technical interview questions that are:
1. Appropriate for a {level} candidate
2. Focused on {topic}
3. Progressive in difficulty
4. Practical and relevant to real-world scenarios
5. Mix of conceptual and application-based questions

IMPORTANT CONSTRAINTS:
- Do NOT ask questions that require writing code or functions
- Do NOT ask to implement algorithms or write programs
- Focus on EXPLAINING concepts, discussing approaches, and verbal reasoning
- Questions should be answerable through spoken explanations only
- Ask about "how would you approach", "explain the concept", "what are the differences", etc.
- Avoid questions starting with "Write", "Implement", "Code", "Create a function"

Format: Return ONLY the questions, one per line, numbered 1-{num_questions}.
Do not include any other text, explanations, or formatting."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Generate {num_questions} {topic} interview questions for {level} candidate.")
    ]
    
    response = llm(messages)
    
    # Parse questions from response
    questions = []
    for line in response.content.strip().split('\n'):
        line = line.strip()
        if line:
            # Remove numbering if present (e.g., "1. ", "1) ", etc.)
            import re
            cleaned = re.sub(r'^\d+[\.)]\s*', '', line)
            if cleaned:
                questions.append(cleaned)
    
    # Ensure we have the requested number of questions
    return questions[:num_questions]


@app.post("/api/interview/start")
async def start_interview(config: InterviewConfig):
    try:
        session_id = str(uuid.uuid4())
        
        # Generate questions dynamically based on topic and experience
        questions = generate_questions(
            config.topic, 
            config.experience_years, 
            config.num_questions
        )
        
        if not questions:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate questions. Please try again."
            )
        
        session = InterviewSession(
            session_id=session_id,
            topic=config.topic,
            experience_years=config.experience_years,
            questions=questions
        )
        interview_sessions[session_id] = session

        return {
            "session_id": session_id,
            "topic": config.topic,
            "experience_years": config.experience_years,
            "total_questions": len(questions),
        }
    except Exception as e:
        print(f"Error starting interview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/interview/{session_id}/question")
async def get_current_question(session_id: str):
    if session_id not in interview_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = interview_sessions[session_id]
    if session.current_question_index >= len(session.questions):
        return {"completed": True, "message": "Interview completed!"}

    current_question = session.questions[session.current_question_index]

    return {
        "completed": False,
        "question": current_question,
        "question_number": session.current_question_index + 1,
        "total_questions": len(session.questions),
        "audio_url": f"/api/interview/{session_id}/audio/question",
    }


@app.get("/api/interview/{session_id}/audio/question")
async def get_question_audio(session_id: str):
    if session_id not in interview_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = interview_sessions[session_id]

    if session.current_question_index >= len(session.questions):
        raise HTTPException(status_code=404, detail="No more questions")

    current_question = session.questions[session.current_question_index]

    response = openai.audio.speech.create(
        model="tts-1", voice="alloy", input=current_question
    )

    return StreamingResponse(io.BytesIO(response.content), media_type="audio/mpeg")


@app.post("/api/interview/{session_id}/answer")
async def process_answer(session_id: str, audio: UploadFile = File(...)):
    try:
        if session_id not in interview_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session = interview_sessions[session_id]
        audio_content = await audio.read()

        # Transcribe audio
        transcription = openai.audio.transcriptions.create(
            model="whisper-1", file=("audio.webm", audio_content, "audio/webm")
        )

        user_answer = transcription.text
        current_question = session.questions[session.current_question_index]

        # Determine experience level context
        if session.experience_years == 0:
            exp_context = "entry-level candidate (fresher)"
        elif session.experience_years <= 2:
            exp_context = f"junior candidate with {session.experience_years} years of experience"
        elif session.experience_years <= 5:
            exp_context = f"mid-level candidate with {session.experience_years} years of experience"
        else:
            exp_context = f"senior candidate with {session.experience_years} years of experience"

        # Evaluate answer
        llm = ChatOpenAI(model="gpt-4", temperature=0.7)

        system_prompt = f"""You are an experienced technical interviewer conducting a {session.topic} interview for a {exp_context}.

Evaluate the candidate's answer considering their experience level.

If the answer is complete and demonstrates good understanding for their level:
- Start your response with EXACTLY "NEXT_QUESTION" on its own line
- Then provide brief positive feedback (1-2 sentences)

If the answer is incomplete or needs clarification:
- Ask ONE specific follow-up question to probe deeper
- Keep it concise and focused
- Adjust difficulty based on their experience level

Current question: {current_question}
Follow-ups asked so far: {session.follow_up_count}/2"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Candidate's answer: {user_answer}"),
        ]
        ai_response = llm(messages)
        response_text = ai_response.content

        # Determine if we should move to next question
        should_move_next = (
            "NEXT_QUESTION" in response_text or session.follow_up_count >= 2
        )

        if should_move_next:
            # Save conversation history for current question
            session.conversation_history.append(
                {
                    "question": current_question,
                    "answer": user_answer,
                    "feedback": response_text.replace("NEXT_QUESTION", "").strip(),
                }
            )

            # Move to next question
            session.current_question_index += 1
            session.follow_up_count = 0

            # Check if interview is complete
            is_complete = session.current_question_index >= len(session.questions)

            # Prepare feedback
            if is_complete:
                feedback = "Excellent work! You've completed all the questions. Let me prepare your interview summary now."
            else:
                feedback = "Great answer! Let's move to the next question."

            # Generate feedback audio
            audio_response = openai.audio.speech.create(
                model="tts-1", voice="alloy", input=feedback
            )

            return StreamingResponse(
                io.BytesIO(audio_response.content),
                media_type="audio/mpeg",
                headers={
                    "X-Should-Load-Next": "true",
                    "X-Interview-Complete": str(is_complete).lower(),
                    "X-Transcription": user_answer,
                    "X-Current-Question-Index": str(session.current_question_index),
                    "X-Total-Questions": str(len(session.questions)),
                },
            )
        else:
            # This is a follow-up
            session.follow_up_count += 1
            feedback = response_text

            # Generate feedback audio
            audio_response = openai.audio.speech.create(
                model="tts-1", voice="alloy", input=feedback
            )

            return StreamingResponse(
                io.BytesIO(audio_response.content),
                media_type="audio/mpeg",
                headers={
                    "X-Should-Load-Next": "false",
                    "X-Interview-Complete": "false",
                    "X-Transcription": user_answer,
                    "X-Follow-Up-Count": str(session.follow_up_count),
                },
            )
    except Exception as e:
        print(f"Error in process_answer: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/interview/{session_id}/summary")
async def get_interview_summary(session_id: str):
    if session_id not in interview_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = interview_sessions[session_id]
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)

    conversation_summary = "\n\n".join(
        [
            f"Q: {item['question']}\nA: {item['answer']}\nFeedback: {item['feedback']}"
            for item in session.conversation_history
        ]
    )

    # Experience level context for summary
    if session.experience_years == 0:
        exp_level = "entry-level/fresher"
    elif session.experience_years <= 2:
        exp_level = "junior (0-2 years)"
    elif session.experience_years <= 5:
        exp_level = "mid-level (3-5 years)"
    else:
        exp_level = "senior (5+ years)"

    system_prompt = f"""Provide a constructive summary of the candidate's {session.topic} interview.
Candidate Experience Level: {exp_level} ({session.experience_years} years)

Include:
1. Overall performance rating considering their experience level (e.g., "Strong", "Good", "Needs Improvement")
2. Key strengths demonstrated
3. Areas for improvement specific to their level
4. Specific topics to study further
5. Career advice and next steps appropriate for their experience
6. Encouraging final remarks

Keep it concise and actionable. Adjust expectations based on their experience level."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Interview conversation:\n\n{conversation_summary}"),
    ]
    summary = llm(messages)

    return {
        "session_id": session_id,
        "topic": session.topic,
        "experience_years": session.experience_years,
        "questions_completed": len(session.conversation_history),
        "summary": summary.content,
        "conversation_history": session.conversation_history,
    }


@app.get("/api/topics")
async def get_available_topics():
    return {
        "topics": [
            "React",
            "JavaScript",
            "Next.js",
            "Python",
            "Node.js",
            "TypeScript",
            "System Design",
            "Data Structures & Algorithms",
            "SQL & Databases",
            "Vue.js",
            "Angular",
            "AWS",
            "DevOps",
            "Machine Learning"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)