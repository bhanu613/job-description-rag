import time
import streamlit as st
from rag import AnswerResult, generate_answer, retrieve

st.set_page_config(page_title="Job Description RAG", layout="centered")
st.title("Job Description RAG")
st.caption("Ask questions about 40 saved job descriptions.")

with st.form("question_form"):
    question = st.text_input("Your question", placeholder="Include the role title when asking about a job.")
    submitted = st.form_submit_button("Ask")

if submitted:
    st.session_state.pop("last_result", None)
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        start, passages = time.perf_counter(), []
        try:
            with st.spinner("Searching and answering..."):
                passages = retrieve(question.strip(), top_k=3)
                response = generate_answer(question.strip(), passages)
        except Exception as error:
            response = AnswerResult("service_error", "The request could not be completed.",
                                    diagnostics=[{"reason": f"{type(error).__name__}: {error}"}])
        st.session_state["last_result"] = {
            "question": question.strip(), "response": response.to_dict(),
            "passages": passages, "seconds": time.perf_counter() - start,
        }

result = st.session_state.get("last_result")
if result:
    response = result["response"]
    st.subheader("Answer")
    st.caption(result["question"])
    if response["status"] == "answered":
        st.markdown(response["answer"].replace("$", "\\$"))
        st.caption("References: " + ", ".join(response["source_ids"]))
    elif response["status"] in {"insufficient_evidence", "no_context"}:
        st.write(response["answer"])
    elif response["status"] == "service_error":
        st.error(response["answer"] + " Check Ollama and the local index.")
    else:
        st.warning(response["answer"])
    st.caption("Generated answers may contain errors. Check the sources.")
    st.subheader("Sources")
    for passage in result["passages"]:
        with st.expander(f"{passage['chunk_id']} | {passage['metadata']['title']}"):
            st.text(passage["text"])
    if not result["passages"]:
        st.write("No passages are available.")
    with st.expander("Run details"):
        st.write(f"Time: {result['seconds']:.1f} seconds")
        st.write(f"Status: {response['status']}")
        if response["status"] in {"validation_failed", "service_error"}:
            st.json(response["diagnostics"])
