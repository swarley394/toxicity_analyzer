import streamlit as st
from detoxify import Detoxify
from textblob import TextBlob

# Custom thresholds (converted to percentages)
TOXICITY_THRESHOLD = 20  # 0.2 → 20%
NEGATIVE_SENTIMENT_THRESHOLD = -30  # -0.3 → -30%

def check_toxicity_detoxify(text):
    """Check toxicity levels using Detoxify and return percentages."""
    results = Detoxify('original').predict(text)
    return {key: value * 100 for key, value in results.items()}  # Convert to %

def check_sentiment_textblob(text):
    """Analyze sentiment using TextBlob and return percentage."""
    return TextBlob(text).sentiment.polarity * 100  # Convert to %

def analyze_post(post):
    """Analyze the post for toxicity and sentiment."""
    detoxify_results = check_toxicity_detoxify(post)
    sentiment_score = check_sentiment_textblob(post)

    is_toxic = any(value > TOXICITY_THRESHOLD for value in detoxify_results.values())
    is_negative = sentiment_score < NEGATIVE_SENTIMENT_THRESHOLD

    status = "🚨 Inappropriate Post!" if is_toxic or is_negative else "✅ Post is Appropriate."

    return detoxify_results, sentiment_score, status

# Streamlit UI
st.title("Bizicard Post Analyzer")
st.write("Enter your post below and analyze its appropriateness.")

post = st.text_area("Post Content", height=200)

if st.button("Check Appropriateness"):
    if post.strip():
        detoxify_results, sentiment_score, status = analyze_post(post)
        
        # Display results
        st.subheader(status)
        st.write(f"**Sentiment Score:** {sentiment_score:.2f}% (Negative if < -30%)")
        
        st.subheader("Detoxify Results (in %)")
        for key, value in detoxify_results.items():
            st.write(f"**{key.capitalize()}**: {value:.2f}%")
    else:
        st.warning("Please enter a post before checking!")
