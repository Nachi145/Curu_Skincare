# Curu_Skincare

Process starts with scrapping reviews from amazon using selenium python library and data is stored as json file. VADER sentiment analysis is run on this json file and result is vader_sentiment_output.json.
Data summarization is done on this json file to get product-level details, stored as a csv file.
Finally, main app is built using streamlit python library.
A separate .env file would be required containing chatGPT API key to be used by streamlit app.
