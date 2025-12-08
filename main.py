from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import io
import os
import uuid
from dotenv import load_dotenv
from openai import OpenAI
import json
import re

# Load environment variables
load_dotenv()

# Ensure key exists
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not set in environment")

client = OpenAI()

app = FastAPI(title="AI Mock Interview API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ai-mock-interview-voice-ui.vercel.app",
        "*"  
    ],
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
        "X-Repeat-Question",
    ],
)

interview_sessions = {}


class TopicConfig(BaseModel):
    tech: str
    subtopics: List[str] = []
    questions: int = 1


class InterviewConfig(BaseModel):
    experience_years: int = 0
    topics: List[TopicConfig]


def generate_questions_multi_tech(topics: List[TopicConfig], experience_years: int):
    """
    Generate questions for multiple technologies with subtopics
    """
    level = (
        "entry-level or fresher" if experience_years == 0 else
        "junior (0-2 years)" if experience_years <= 2 else
        "mid-level (3-5 years)" if experience_years <= 5 else
        "senior (5+ years)"
    )

    difficulty = (
        "basic concepts" if experience_years == 0 else
        "intermediate concepts" if experience_years <= 2 else
        "advanced architectural concepts" if experience_years <= 5 else
        "expert-level design & leadership concepts"
    )

    all_questions = []

    for topic_config in topics:
        tech = topic_config.tech
        subtopics = topic_config.subtopics
        num_questions = topic_config.questions

        # Build subtopic instruction
        subtopic_instruction = ""
        if subtopics:
            subtopic_instruction = f"\nFocus specifically on these subtopics: {', '.join(subtopics)}"

        system_prompt = f"""
You are an experienced interviewer.

Generate exactly {num_questions} interview questions.

Technology: {tech}{subtopic_instruction}
Experience Level: {level}
Focus Difficulty: {difficulty}

Rules:
- Only verbal explanation questions.
- No coding tasks.
- Use numbering (1., 2., ...).
- Return ONLY the numbered questions, no extra commentary.
- Questions should be relevant to the technology and subtopics specified.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate {num_questions} questions about {tech}."}
            ],
        )

        raw = response.choices[0].message.content

        questions = []
        for line in raw.split("\n"):
            line = line.strip()
            if line:
                cleaned = re.sub(r"^\d+[\.)]\s*", "", line)
                if cleaned:
                    questions.append(cleaned)

        # Take only the requested number
        questions = questions[:num_questions]

        # Add to all_questions with metadata
        for q in questions:
            all_questions.append({
                "text": q,
                "tech": tech,
                "subtopics": subtopics
            })

    return all_questions


@app.post("/api/interview/start")
async def start_interview(config: InterviewConfig):
    try:
        if not config.topics or len(config.topics) == 0:
            raise HTTPException(400, "At least one topic must be selected")

        session_id = str(uuid.uuid4())
        
        # Generate questions for all technologies
        questions = generate_questions_multi_tech(config.topics, config.experience_years)

        session = InterviewSession(
            session_id=session_id,
            experience_years=config.experience_years,
            topics=config.topics,
            questions=questions
        )

        interview_sessions[session_id] = session

        # Build topics summary for response
        topics_summary = ", ".join([t.tech for t in config.topics])

        return {
            "session_id": session_id,
            "topics": topics_summary,
            "experience_years": config.experience_years,
            "total_questions": len(questions),
        }

    except Exception as e:
        print("Error starting interview:", e)
        raise HTTPException(500, str(e))


@app.get("/api/interview/{session_id}/question")
async def get_question(session_id: str):
    if session_id not in interview_sessions:
        raise HTTPException(404, "Session not found")

    session = interview_sessions[session_id]

    if session.current_question_index >= len(session.questions):
        return {"completed": True, "message": "Interview completed!"}

    current_q = session.questions[session.current_question_index]

    return {
        "completed": False,
        "question": current_q["text"],
        "tech": current_q["tech"],
        "subtopics": current_q["subtopics"],
        "question_number": session.current_question_index + 1,
        "total_questions": len(session.questions),
        "audio_url": f"/api/interview/{session_id}/audio/question",
    }


@app.get("/api/interview/{session_id}/audio/question")
async def question_audio(session_id: str):
    if session_id not in interview_sessions:
        raise HTTPException(404, "Session not found")

    session = interview_sessions[session_id]

    if session.current_question_index >= len(session.questions):
        raise HTTPException(404, "No more questions")

    text = session.questions[session.current_question_index]["text"]

    audio = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )

    return StreamingResponse(io.BytesIO(audio.content), media_type="audio/mpeg")


def safe_parse_json(text: str):
    """
    Try to parse JSON from the model output.
    """
    text = text.strip()
  
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if m:
            candidate = m.group(1)
            try:
                return json.loads(candidate)
            except Exception:
                pass
        m2 = re.search(r"(\[.*\])", text, flags=re.DOTALL)
        if m2:
            candidate = m2.group(1)
            try:
                return json.loads(candidate)
            except Exception:
                pass
    raise ValueError("No valid JSON found")

# 1.# Key changes to make in your code:

# 1. Update InterviewSession model to track pending follow-up
class InterviewSession(BaseModel):
    session_id: str
    experience_years: int
    topics: List[TopicConfig]
    questions: List[dict]
    current_question_index: int = 0
    conversation_history: List[dict] = []
    follow_up_count: int = 0
    # NEW: Track if we're in follow-up mode and what was asked
    pending_follow_up: Optional[str] = None  # Stores the follow-up question text
    accumulated_answer: str = ""  # Accumulates answers for current question
    wrong_attempt_count: int = 0  # Track wrong attempts for current question/follow-up


# 2. Update the process_answer endpoint - replace the evaluation section:

@app.post("/api/interview/{session_id}/answer")
async def process_answer(session_id: str, audio: UploadFile = File(...)):
    try:
        if session_id not in interview_sessions:
            raise HTTPException(404, "Session not found")

        session = interview_sessions[session_id]
        content = await audio.read()

        # Transcribe audio
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=("audio.webm", content, "audio/webm")
        )
        user_answer = transcript.text

        current_q = session.questions[session.current_question_index]
        current_question = current_q["text"]
        current_tech = current_q["tech"]

        exp_level = (
            "entry-level (fresher)" if session.experience_years == 0 else
            f"junior ({session.experience_years} years)" if session.experience_years <= 2 else
            f"mid-level ({session.experience_years} years)" if session.experience_years <= 5 else
            f"senior ({session.experience_years} years)"
        )

        # Accumulate the answer
        if session.accumulated_answer:
            session.accumulated_answer += " " + user_answer
        else:
            session.accumulated_answer = user_answer

        # Determine what question to evaluate against
        if session.pending_follow_up:
            # We're evaluating a follow-up response
            evaluation_question = session.pending_follow_up
            evaluation_context = f"This is a follow-up response. Original question was: '{current_question}'. Follow-up asked: '{session.pending_follow_up}'"
        else:
            # First response to the main question
            evaluation_question = current_question
            evaluation_context = f"This is the initial response to the main question."

        system_prompt = f"""You are an interviewer evaluating a {exp_level} candidate on {current_tech}.
You must respond ONLY with a single valid JSON object (no extra commentary).

Context: {evaluation_context}

JSON schema:
{{
  "status": "WRONG" | "PARTIAL" | "CORRECT",
  "message": "string (short explanation / feedback)",
  "follow_up_question": "string or null",
  "next_question": "string or null"
}}

Rules:
- Evaluate ONLY the candidate's response to the specific question being asked.
- If evaluating a follow-up, judge whether they answered THAT follow-up adequately.
- If the answer is totally incorrect or irrelevant → status = "WRONG".
  Set message to ONLY "That answer is not correct. Please think again and try once more." 
  Do NOT give any hints, explanations, or correct answer information.
  Set follow_up_question=null, next_question=null.
- If the answer is partially correct or incomplete → status = "PARTIAL".
  Set message and set follow_up_question to a single concise follow-up question (string).
  next_question must be null.
- If the answer is fully correct → status = "CORRECT".
  Set message (short positive feedback). You may optionally include next_question (string) but it's not required.

Be careful: return only valid JSON."""

        # Simple evaluation with just the question and answer
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {evaluation_question}\nAnswer: {user_answer}"}
            ],
        )

        model_output = response.choices[0].message.content.strip()

        try:
            eval_json = safe_parse_json(model_output)
        except Exception as parse_err:
            print("Model JSON parse failed:", parse_err)
            eval_json = {
                "status": "WRONG",
                "message": "I couldn't reliably evaluate that answer; please try again.",
                "follow_up_question": None,
                "next_question": None
            }

        status = eval_json.get("status", "").upper()
        message = eval_json.get("message", "").strip()
        follow_up_q = eval_json.get("follow_up_question")
        next_q = eval_json.get("next_question")

        # Handle WRONG answer
        if status == "WRONG":
            session.wrong_attempt_count += 1
            
            # After 2 wrong attempts, move to next question regardless
            if session.wrong_attempt_count >= 2:
                session.conversation_history.append({
                    "question": current_question,
                    "tech": current_tech,
                    "answer": session.accumulated_answer,
                    "feedback": "Multiple incorrect attempts. Moving to next question."
                })

                session.current_question_index += 1
                session.follow_up_count = 0
                session.pending_follow_up = None
                session.accumulated_answer = ""
                session.wrong_attempt_count = 0  # Reset for next question

                speak_text = "Let's move to the next question."
                audio_resp = client.audio.speech.create(model="tts-1", voice="alloy", input=speak_text)

                interview_complete = session.current_question_index >= len(session.questions)

                return StreamingResponse(
                    io.BytesIO(audio_resp.content),
                    media_type="audio/mpeg",
                    headers={
                        "X-Should-Load-Next": "true",
                        "X-Interview-Complete": str(interview_complete).lower(),
                        "X-Transcription": user_answer,
                        "X-Follow-Up-Count": "0",
                    },
                )
            
            # First wrong attempt - give them another chance
            # Keep pending_follow_up so they retry the SAME question
            speak_text = f"{message or 'That answer was not correct. Please try again.'} Let me repeat the question: {evaluation_question}"

            audio_resp = client.audio.speech.create(model="tts-1", voice="alloy", input=speak_text)

            return StreamingResponse(
                io.BytesIO(audio_resp.content),
                media_type="audio/mpeg",
                headers={
                    "X-Should-Load-Next": "false",
                    "X-Interview-Complete": "false",
                    "X-Transcription": user_answer,
                    "X-Follow-Up-Count": str(session.follow_up_count),
                    "X-Repeat-Question": "true",
                },
            )

        # Handle PARTIAL answer
        if status == "PARTIAL":
            session.follow_up_count += 1
            session.wrong_attempt_count = 0  # Reset wrong attempts on partial (they're making progress)

            # After 2 follow-ups, move to next question
            if session.follow_up_count >= 2:
                session.conversation_history.append({
                    "question": current_question,
                    "tech": current_tech,
                    "answer": session.accumulated_answer,
                    "feedback": message or "Partial answer (follow-ups completed)."
                })

                session.current_question_index += 1
                session.follow_up_count = 0
                session.pending_follow_up = None
                session.accumulated_answer = ""
                session.wrong_attempt_count = 0  # Reset for next question

                speak_text = "Thanks. Let's move to the next question."
                audio_resp = client.audio.speech.create(model="tts-1", voice="alloy", input=speak_text)

                interview_complete = session.current_question_index >= len(session.questions)

                return StreamingResponse(
                    io.BytesIO(audio_resp.content),
                    media_type="audio/mpeg",
                    headers={
                        "X-Should-Load-Next": "true",
                        "X-Interview-Complete": str(interview_complete).lower(),
                        "X-Transcription": user_answer,
                        "X-Follow-Up-Count": "0",
                    },
                )

            # Ask follow-up and store it
            follow_up_text = follow_up_q or "Could you clarify that part in more detail?"
            session.pending_follow_up = follow_up_text  # Store the follow-up question
            
            speak_text = f"{message}. {follow_up_text}"

            audio_resp = client.audio.speech.create(model="tts-1", voice="alloy", input=speak_text)

            return StreamingResponse(
                io.BytesIO(audio_resp.content),
                media_type="audio/mpeg",
                headers={
                    "X-Should-Load-Next": "false",
                    "X-Interview-Complete": "false",
                    "X-Transcription": user_answer,
                    "X-Follow-Up-Count": str(session.follow_up_count),
                },
            )

        # Handle CORRECT answer
        if status == "CORRECT":
            session.conversation_history.append({
                "question": current_question,
                "tech": current_tech,
                "answer": session.accumulated_answer,
                "feedback": message or "Good answer."
            })

            session.current_question_index += 1
            session.follow_up_count = 0
            session.pending_follow_up = None
            session.accumulated_answer = ""
            session.wrong_attempt_count = 0  # Reset for next question

            interview_complete = session.current_question_index >= len(session.questions)

            speak_text = message or ("Great! Moving to the next question." if not interview_complete else "You have completed the interview. Preparing summary.")

            audio_resp = client.audio.speech.create(model="tts-1", voice="alloy", input=speak_text)

            return StreamingResponse(
                io.BytesIO(audio_resp.content),
                media_type="audio/mpeg",
                headers={
                    "X-Should-Load-Next": "true",
                    "X-Interview-Complete": str(interview_complete).lower(),
                    "X-Transcription": user_answer,
                },
            )

        # Fallback
        speak_text = "I couldn't interpret the evaluation. Please try answering again."
        audio_resp = client.audio.speech.create(model="tts-1", voice="alloy", input=speak_text)
        return StreamingResponse(
            io.BytesIO(audio_resp.content),
            media_type="audio/mpeg",
            headers={
                "X-Should-Load-Next": "false",
                "X-Interview-Complete": "false",
                "X-Transcription": user_answer,
                "X-Follow-Up-Count": str(session.follow_up_count),
                "X-Repeat-Question": "true",
            },
        )

    except Exception as e:
        print("Error in process_answer:", e)
        raise HTTPException(500, str(e))

@app.get("/api/interview/{session_id}/summary")
async def get_summary(session_id: str):
    if session_id not in interview_sessions:
        raise HTTPException(404, "Session not found")

    session = interview_sessions[session_id]

    summary_text = ""
    for item in session.conversation_history:
        summary_text += f"Tech: {item.get('tech', 'General')}\nQ: {item['question']}\nA: {item['answer']}\nFeedback: {item['feedback']}\n\n"

    exp_level = (
        "entry-level/fresher" if session.experience_years == 0 else
        "junior" if session.experience_years <= 2 else
        "mid-level" if session.experience_years <= 5 else
        "senior"
    )

    # Get all technologies covered
    techs_covered = list(set([t.tech for t in session.topics]))

    system_prompt = f"""
Provide a constructive interview summary for a {exp_level} candidate.
Technologies covered: {', '.join(techs_covered)}

Include:
- Overall performance rating
- Strengths shown across different technologies
- Weaknesses and areas needing improvement
- Specific topics to focus on for each technology
- Encouraging closing notes

Keep it concise but comprehensive.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": summary_text}
        ],
    )

    summary = response.choices[0].message.content

    return {
        "session_id": session_id,
        "technologies": techs_covered,
        "experience_years": session.experience_years,
        "questions_completed": len(session.conversation_history),
        "summary": summary,
        "conversation_history": session.conversation_history
    }


@app.get("/api/topics")
async def get_topics():
    return {
"topics": [
            "React",
            "JavaScript",
            "Next.js",
            "React Native",
            "Python",
            "FastAPI",
            "Node.js",
            "Express",
            "Nest",
            "MongoDB",
            "Agile",
            "GIT",
            "TypeScript",
            "Supabase",
            "Firebase",
            "System Design",
            "SQL & Databases",
            "Vue.js",
            "Angular",
            "AWS",
            "DevOps",
            "Machine Learning",
            "Docker",
            "Kubernetes",
            "AI Agent",
            "Multi-Agent Systems",
            "LangChain",
            "LangGraph",
            "Pinecone",
            "Weaviate",
            "Chroma",
            "Milvus",
            "FAISS",
            "Redis Vector DB"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)