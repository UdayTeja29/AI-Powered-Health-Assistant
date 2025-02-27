import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load DistilGPT-2 model and tokenizer
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure the model can handle padding
tokenizer.pad_token = tokenizer.eos_token

def chatbot_response(user_input):
    # Format input
    prompt = f"User: {user_input}\nChatbot:"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)

    # Generate response
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=150,
        num_beams=5,
        no_repeat_ngram_size=3,
        top_p=0.9,
        temperature=0.7,
        do_sample=True,
        early_stopping=True
    )

    # Decode and clean response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    bot_response = response.replace(prompt, "").strip()

    if len(bot_response) < 5:  # Prevent empty/irrelevant output
        return "I'm here to help! Could you rephrase your question?"

    return bot_response

# Streamlit UI
def main():
    st.title("Healthcare Chatbot")
    st.write("Ask me anything!")

    user_input = st.text_input("Your question:")

    if st.button("Submit"):
        if user_input:
            st.write("User:", user_input)
            
            with st.spinner("Thinking..."):
                response = chatbot_response(user_input)
            
            st.write("Chatbot:", response)
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
