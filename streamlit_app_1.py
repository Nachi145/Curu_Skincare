import streamlit as st
import pandas as pd
import json
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import re
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go

from collections import defaultdict

# Load environment variables for OpenAI
load_dotenv()

print("API KEY:", os.getenv("OPENAI_API_KEY"))

# Load data
@st.cache_data
def load_data():
    with open("vader_sentiment_output.json", "r") as f:
        reviews = json.load(f)
    summary_df = pd.read_csv("product_vader_scores.csv")
    return reviews, summary_df

reviews, summary_df = load_data()

# Sidebar for product selection
st.title("🧴 Product Review Analysis (Sentiments + Review Insights)")
product_list = summary_df["product"].tolist()
selected_product = st.selectbox("Select a Product", product_list)

# Filter reviews
product_reviews = [r for r in reviews if r["product"] == selected_product]
positive_reviews = [r["review"] for r in product_reviews if r["vader_sentiment"] == "Positive"]
negative_reviews = [r["review"] for r in product_reviews if r["vader_sentiment"] == "Negative"]

# Fetchning and combining equal number of positive and negative reviews to get balanced reviews
positive_reviews = positive_reviews[:50]
negative_reviews = negative_reviews[:50]
bal_reviews = positive_reviews + negative_reviews

# Get summary row
product_stats = summary_df[summary_df["product"] == selected_product].iloc[0]

# Display summary
st.subheader("📊 Sentiment Overview")

sentiment_color = {
    "Positive": "green",
    "Negative": "red",
    "Neutral": "gray"
}

overall = product_stats["overall_sentiment"]

st.markdown(f"**Overall Sentiment:** <span style='color:{sentiment_color[overall]}; font-size:20px;'>{overall}</span>", unsafe_allow_html=True)

compound_score = product_stats["avg_compound"]  # e.g. -0.3 to 0.9

gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=compound_score,
    title={'text': "Compound Sentiment Score"},
    gauge={
        'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "darkgray"},
        'bar': {'color': "blue"},
        'steps': [
            {'range': [-1, -0.5], 'color': "red"},
            {'range': [-0.5, 0], 'color': "orange"},
            {'range': [0, 0.5], 'color': "lightgreen"},
            {'range': [0.5, 1], 'color': "green"},
        ],
        'threshold': {
            'line': {'color': "black", 'width': 4},
            'thickness': 0.75,
            'value': compound_score
        }
    }
))

gauge.update_layout(height=150, margin=dict(t=10, b=10, l=10, r=10))
st.plotly_chart(gauge, use_container_width=True)

# # Compound score as progress bar (-1 to 1 → normalized to 0 to 1)
# normalized_compound = (product_stats["avg_compound"] + 1) / 2
# st.write("**Sentiment Polarity (Compound Score)**")
# st.progress(normalized_compound)

# # Positive/Negative score bars
# st.write("**Average Sentiment Breakdown**")
# sentiment_scores = {
#     "Positive": product_stats["avg_pos"],
#     "Negative": product_stats["avg_neg"]
# }
# for label, score in sentiment_scores.items():
#     st.write(f"{label} Score: {score:.2f}")
#     st.progress(score)
sentiment_counts = {
    "Positive": product_stats["positive_reviews"],
    "Negative": product_stats["negative_reviews"],
    "Neutral": product_stats["neutral_reviews"]
}
bar_df = pd.DataFrame(list(sentiment_counts.items()), columns=["Sentiment", "Count"])
sentiment_order = ["Positive", "Neutral", "Negative"]
bar_df["Sentiment"] = pd.Categorical(bar_df["Sentiment"], categories=sentiment_order, ordered=True)
bar_df = bar_df.sort_values("Sentiment", ascending=True)
#bar_df = bar_df[bar_df["Count"] > 0]
fig = px.bar(
    bar_df,
    x="Count",
    y="Sentiment",
    orientation="h",
    color="Sentiment",
    color_discrete_map={"Positive": "green", "Negative": "red", "Neutral": "gray"},
    text="Count",
)

fig.update_traces(
    textposition="outside",
    textfont_size=16,
    marker_line_width=1.2
)

fig.update_layout(
    title="Review Sentiment Distribution",
    title_font_size=16,
    xaxis_title="Number of Reviews",
    yaxis_title="Sentiment",
    yaxis_tickfont_size=14,
    xaxis_tickfont_size=14,
    height=300,
    bargap=0.35,        # smaller gaps between bars (default 0.2)
    bargroupgap=0.05,
    margin=dict(t=30, b=20, l=10, r=10),
    showlegend=False,
    uniformtext_minsize=12,
    uniformtext_mode='hide'
)

st.plotly_chart(fig, use_container_width=True)

# Prompt template
review_analysis_prompt = PromptTemplate(
    input_variables=["reviews"],
    template="""
    You are an expert assistant analyzing customer reviews for a skincare product.

    Given the following customer reviews (mixed positive and negative), do the following:
    1. Identify the **top 5 positive key points** customers appreciated.
    2. Identify the **top 5 negative key points** customers complained about.
    3. Extract up to **8 important keywords** (relevant to skincare concerns, ingredients, 
    skin types, product effects, or common themes) mentioned in the reviews that customers look for. For each keyword, include:
    - Total number of **positive mentions**
    - Total number of **negative mentions**
    - Ensure that similar keywords (like "oily skin" and "greasy") are considered together.
    
    Reviews:
    {reviews}

    Output format:
    ### Positive Insights:
    - ...
    - ...

    ### Negative Insights:
    - ...
    - ...

    ### Top Keywords and Mentions:
    - keyword1: X positive mentions, Y negative mentions  
    - keyword2: A positive mentions, B negative mentions  
    ... up to 8
    """
)

llm = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo")
chain = LLMChain(llm=llm, prompt=review_analysis_prompt)

with st.spinner("Analyzing reviews..."):
    response = chain.run({"reviews": " ".join(bal_reviews)})

st.subheader("📈 Review Analysis Summary")
#st.markdown(response)

positive_section = re.search(r"### Positive Insights:\n(.*?)\n###", response, re.DOTALL)
negative_section = re.search(r"### Negative Insights:\n(.*?)\n###", response, re.DOTALL)
keywords_section = re.search(r"### Top Keywords and Mentions:\n(.*)", response, re.DOTALL)

st.subheader("✨ Positive Insights")
st.success(positive_section.group(1).strip() if positive_section else "Not found.")

st.subheader("⚠️ Negative Insights")
st.error(negative_section.group(1).strip() if negative_section else "Not found.")

st.subheader("🔑 Top Keywords")
st.markdown(keywords_section.group(1).strip() if keywords_section else "Not found.")

# # Generate insights
# if positive_reviews:
#     with st.spinner("Extracting positive insights..."):
#         pos_response = chain.run({"sentiment": "positive", "reviews": " ".join(positive_reviews[:40])})

# if negative_reviews:
#     with st.spinner("Extracting negative insights..."):
#         neg_response = chain.run({"sentiment": "negative", "reviews": " ".join(negative_reviews[:40])})

# # Display insights
# st.subheader("✨ Key Positive Insights")
# if positive_reviews:
#     st.success(pos_response)
# else:
#     st.info("No positive reviews available.")

# st.subheader("⚠️ Key Negative Insights")
# if negative_reviews:
#     st.error(neg_response)
# else:
#     st.info("No negative reviews available.")

# keywords = [
#     "dry skin", "oily skin", "combination skin",
#     "moisturizing", "hydrating", "fragrance",
#     "gentle", "non-comedogenic", "irritation"
# ]

# def count_keyword_sentiment(reviews, keywords):
#     counts = defaultdict(lambda: {"Positive": 0, "Negative": 0})
#     for r in reviews:
#         sentiment = r["vader_sentiment"]
#         if sentiment not in ["Positive", "Negative"]:
#             continue
#         text = r["review"].lower()
#         for kw in keywords:
#             if kw in text:
#                 counts[kw][sentiment] += 1
#     return counts

# sentiment_counts = count_keyword_sentiment(product_reviews, keywords)

# df_sentiment = pd.DataFrame([
#     {"Keyword": kw, "Sentiment": "Positive", "Count": c["Positive"]}
#     for kw, c in sentiment_counts.items()
# ] + [
#     {"Keyword": kw, "Sentiment": "Negative", "Count": c["Negative"]}
#     for kw, c in sentiment_counts.items()
# ])

# if not df_sentiment.empty and df_sentiment["Count"].sum() > 0:
#     st.subheader("🧼 Keyword Mentions with Context")

#     fig = px.bar(
#         df_sentiment,
#         x="Count", y="Keyword",
#         color="Sentiment",
#         orientation="h",
#         barmode="stack",
#         color_discrete_map={"Positive": "green", "Negative": "red"},
#         #title="Positive vs Negative Mentions of Key Skincare Terms"
#     )

#     fig.update_layout(height=300, 
#                       margin=dict(t=30, b=20, l=10, r=10),
#                       bargap=0.35,        # smaller gaps between bars (default 0.2)
#                       bargroupgap=0.05,  # smaller space between groups
#                       yaxis=dict(tickfont=dict(size=14)),  # smaller font
#                       xaxis=dict(tickfont=dict(size=14)),
#                       #title_font_size=16,
#                       legend=dict(font=dict(size=14)),)
#     st.plotly_chart(fig, use_container_width=True)
# else:
#     st.info("Keywords not found or no sentiment data available.")