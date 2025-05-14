# Mini-project-1
📌 Title and Objective

**Title**: *What Does the Government Say? Analyzing Sentiment in NYC 311 Resolution Responses Using NLP*

**Objective**:
This project explores the tone of government responses to public service requests submitted through New York City’s 311 system. While most 311 data analyses focus on what issues are being reported or how quickly they are resolved, I wanted to investigate something different: *how* the government communicates its response back to the public. Using Natural Language Processing (NLP), I analyzed the `resolution_description` field—a column that includes official statements issued by agencies to close out complaints.

The core goal was to identify patterns in the sentiment of these responses. I asked questions like: Are some boroughs receiving more negatively worded responses? Are certain complaint types associated with more constructive or dismissive language? This project contributes to the broader theme of data science for public good by offering insights into civic communication, accountability, and equity in government responsiveness.

---

### 📊 Data Sources

* **Source**: NYC Open Data Portal
* **Dataset**: 311 Service Requests (2023 onward)
* **Access Method**: Socrata Open Data API (`erm2-nwe9`)
* **Key Variables Used**:

  * `resolution_description`: text of agency response
  * `borough`: geographic location of the complaint
  * `complaint_type`: type of issue reported
* **Data Size**: 48,749 resolution entries
* **License**: Open Data Commons Public Domain Dedication and License (PDDL)

**Why this dataset?**
I selected this dataset because it offers a large volume of public service interactions that include unstructured text. Unlike many public datasets that are mostly numerical or categorical, this one provided a unique opportunity to work with free-text communication from government employees to citizens. It also aligned perfectly with the "public good" mission of the course, as the content directly reflects city responsiveness and communication with residents.

---

### 🛠️ Methods and Code Snippets

This project was implemented using Python in a Jupyter Notebook. I used several key libraries:

* `pandas` for data wrangling
* `TextBlob` for sentiment analysis
* `matplotlib` and `seaborn` for visualization

#### Step 1: Sentiment Polarity Calculation

First, I used TextBlob to generate sentiment polarity scores for each resolution description:

```python
df['resolution_description'] = df['resolution_description'].astype(str).fillna('')
df['sentiment'] = df['resolution_description'].apply(lambda x: TextBlob(x).sentiment.polarity)
```

The scores range from -1 (very negative) to +1 (very positive).

#### Step 2: Sentiment Classification into Labels

To make the scores more interpretable, I created a function that labeled each score as Positive, Neutral, or Negative:

```python
def classify_sentiment(score):
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment_label'] = df['sentiment'].apply(classify_sentiment)
```

#### Step 3: Visualization by Borough

I visualized sentiment categories by borough using a count plot to see how sentiment distribution varied geographically:

```python
sns.countplot(data=df, x='borough', hue='sentiment_label', palette='Set2')
```

#### Step 4: Complaint Type vs. Average Sentiment

Finally, I grouped data by complaint type and calculated average sentiment to identify types of issues most associated with negative or positive language:

```python
complaint_sentiment = df.groupby('complaint_type')['sentiment'].mean().sort_values()
```

---

### 📈 Key Findings

* **Sentiment Breakdown**:

  * Negative: **25,833** responses
  * Positive: **15,021** responses
  * Neutral: **7,895** responses

* **Borough-Level Trends**:

  * **Staten Island** had the most negative average sentiment.
  * **Manhattan** had the least negative sentiment, leaning more neutral overall.
  * Brooklyn and Queens had large volumes of negative responses, consistent with high 311 activity levels.

* **Most Negatively Responded Complaint Types**:

  * Plant (-0.3167)
  * Snow or Ice (-0.30)
  * Recycling Basket Complaint, LinkNYC, Transfer Station Complaint (all \~ -0.30)

* **Most Positively Responded Complaint Types**:

  * Taxi Compliment (+0.50)
  * Traffic Signal Condition (+0.405)
  * Street Light Condition (+0.403)

These findings suggest that certain complaint types, especially those involving sanitation or environmental issues, tend to receive colder or more dismissive language. Conversely, infrastructure maintenance or compliments tend to receive more constructive and positive responses.

---

### 💬 Reflection

This was one of the most meaningful projects I completed during the semester. It was the first time I applied NLP techniques to real-world, unstructured text with public policy implications. At first, I wasn’t sure whether something as simple as sentiment analysis would reveal anything substantial from the short and formal government response texts. However, by classifying and visualizing sentiment, I found clear trends that connected to geography, service categories, and communication style.

One key learning was the limitation of basic sentiment scoring tools. A phrase like “No further action required” might sound neutral but could feel dismissive to a resident. Future work could involve customizing a sentiment model that is better tuned for public service contexts. I also see potential in extending this analysis to look at time trends, repeat complaints, or even topic modeling (LDA) to extract themes from responses.

This project helped me see the human side of public data—people are not just submitting service requests; they are also receiving answers that impact their perception of government responsiveness. Using data science to evaluate those interactions made this feel both technically rewarding and socially relevant.
