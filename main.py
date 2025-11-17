from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List
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


def generate_questions(topic: str, experience_years: int, num_questions: int):
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

    system_prompt = f"""
You are an experienced interviewer.

Generate exactly {num_questions} interview questions.

Topic: {topic}
Experience Level: {level}
Focus Difficulty: {difficulty}

Rules:
- Only verbal explanation questions.
- No coding tasks.
- Use numbering (1., 2., ...).
- Return ONLY the numbered questions, no extra commentary.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate {num_questions} questions."}
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

    return questions[:num_questions]


@app.post("/api/interview/start")
async def start_interview(config: InterviewConfig):
    try:
        session_id = str(uuid.uuid4())
        questions = generate_questions(config.topic, config.experience_years, config.num_questions)

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
        print("Error starting interview:", e)
        raise HTTPException(500, str(e))



@app.get("/api/interview/{session_id}/question")
async def get_question(session_id: str):
    if session_id not in interview_sessions:
        raise HTTPException(404, "Session not found")

    session = interview_sessions[session_id]

    if session.current_question_index >= len(session.questions):
        return {"completed": True, "message": "Interview completed!"}

    return {
        "completed": False,
        "question": session.questions[session.current_question_index],
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

    text = session.questions[session.current_question_index]

    audio = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )

    return StreamingResponse(io.BytesIO(audio.content), media_type="audio/mpeg")


def safe_parse_json(text: str):
    """
    Try to parse JSON from the model output.
    1) Direct json.loads
    2) Extract first JSON object/array substring via regex and parse
    Returns dict or raises ValueError.
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



@app.post("/api/interview/{session_id}/answer")
async def process_answer(session_id: str, audio: UploadFile = File(...)):
    try:
        if session_id not in interview_sessions:
            raise HTTPException(404, "Session not found")

        session = interview_sessions[session_id]
        content = await audio.read()


        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=("audio.webm", content, "audio/webm")
        )
        user_answer = transcript.text

        current_question = session.questions[session.current_question_index]

        exp_level = (
            "entry-level (fresher)" if session.experience_years == 0 else
            f"junior ({session.experience_years} years)" if session.experience_years <= 2 else
            f"mid-level ({session.experience_years} years)" if session.experience_years <= 5 else
            f"senior ({session.experience_years} years)"
        )


        system_prompt = f"""
You are an interviewer evaluating a {exp_level} candidate.
You must respond ONLY with a single valid JSON object (no extra commentary).

JSON schema:
{{
  "status": "WRONG" | "PARTIAL" | "CORRECT",
  "message": "string (short explanation / feedback)",
  "follow_up_question": "string or null",
  "next_question": "string or null"
}}

Rules:
- If the answer is totally incorrect or irrelevant → status = "WRONG".
  Set message explaining briefly why and set follow_up_question=null, next_question=null.
- If the answer is partially correct → status = "PARTIAL".
  Set message and set follow_up_question to a single concise follow-up question (string).
  next_question must be null.
- If the answer is fully correct → status = "CORRECT".
  Set message (short positive feedback). You may optionally include next_question (string) but it's not required.

Be careful: return only valid JSON. Example:
{{"status":"PARTIAL","message":"Missing key tradeoffs","follow_up_question":"What trade-offs would you consider for scaling?", "next_question":null}}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {current_question}\nAnswer: {user_answer}"}
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


        if status == "WRONG":
            speak_text = f"{message or 'That answer was not correct. Please try again.'} I'll repeat the question: {current_question}"

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


        if status == "PARTIAL":
            
            session.follow_up_count += 1

            
            if session.follow_up_count >= 2:
                
                session.conversation_history.append({
                    "question": current_question,
                    "answer": user_answer,
                    "feedback": message or "Partial answer (follow-ups completed)."
                })

               
                session.current_question_index += 1
                session.follow_up_count = 0

                
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

            
            follow_up_text = follow_up_q or "Could you clarify that part in more detail?"

            
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


        if status == "CORRECT":
            
            session.conversation_history.append({
                "question": current_question,
                "answer": user_answer,
                "feedback": message or "Good answer."
            })

            session.current_question_index += 1
            session.follow_up_count = 0

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
        summary_text += f"Q: {item['question']}\nA: {item['answer']}\nFeedback: {item['feedback']}\n\n"

    exp_level = (
        "entry-level/fresher" if session.experience_years == 0 else
        "junior" if session.experience_years <= 2 else
        "mid-level" if session.experience_years <= 5 else
        "senior"
    )

    system_prompt = f"""
Provide a constructive interview summary for a {exp_level} candidate.
Include:
- Overall performance rating
- Strengths
- Weaknesses
- Topics to improve
- Encouraging closing notes
Keep it short.
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
        "topic": session.topic,
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
