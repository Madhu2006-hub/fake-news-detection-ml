import gradio as gr

def detect_fake_news(news_text):
    vec = vectorizer.transform([news_text])
    prediction = model.predict(vec)[0]

    if prediction == 1:
        return "Real News"
    else:
        return "Fake News"

demo = gr.Interface(
    fn=detect_fake_news,
    inputs=gr.Textbox(lines=6, placeholder="Enter news article here..."),
    outputs="text",
    title="Fake News Detection System"
)

demo.launch()
