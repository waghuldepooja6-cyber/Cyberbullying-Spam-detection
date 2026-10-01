import mysql.connector
from datetime import datetime
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database_v2.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float)
    detection_date = db.Column(db.String(20))
    detection_time = db.Column(db.String(20))

with app.app_context():
    db.create_all() 

# ==========================================================
# 1. TRAINING DATA
# ==========================================================

normal_messages = [
    "hello",
    "hi",
    "hello how are you",
    "hi how are you",
    "i am fine",
    "i am good",
    "i am doing well",
    "good morning",
    "good afternoon",
    "good evening",
    "good night",
    "how are you",
    "what are you doing",
    "nice to meet you",
    "have a nice day",
    "thank you",
    "thanks",
    "welcome",
    "see you tomorrow",
    "lets study together",
    "let's study together",
    "can you help me",
    "please send the notes",
    "what is the homework",
    "i completed my assignment",
    "let us meet tomorrow",
    "where are you going",
    "i like this project",
    "this is a good idea",
    "everything is fine",
    "nothing is wrong",
    "you are my friend",
    "you are a good friend",
    "let us work together",
    "please send me the assignment",
    "can we study together",
    "i will call you later",
    "see you soon",
    "good luck",
    "all the best",
    "well done",
    "that is nice",
    "okay",
    "ok",
    "sure",
    "yes",
    "no problem"
]

spam_messages = [
    "you won a lottery",
    "congratulations you won a prize",
    "click here to win money",
    "claim your free prize",
    "you have won 100000",
    "free money click now",
    "win a free iphone",
    "click this link to get reward",
    "limited time offer",
    "get discount now",
    "you are selected for a cash prize",
    "claim your reward now",
    "free gift waiting for you",
    "earn money quickly",
    "make money from home",
    "special offer click now",
    "buy now and get huge discount",
    "you won a free gift",
    "click to receive your prize",
    "exclusive offer for you",
    "free recharge available",
    "win exciting prizes",
    "congratulations claim your reward",
    "urgent offer click here",
    "click now",
    "free offer",
    "get free money",
    "win cash prize",
    "special discount",
    "claim your gift"
]

cyberbullying_messages = [
    "you are stupid",
    "you are dumb",
    "you are useless",
    "nobody likes you",
    "you are annoying",
    "you are a loser",
    "shut up",
    "you are terrible",
    "you are an idiot",
    "you cannot do anything",
    "everyone hates you",
    "you are so bad",
    "you are worthless",
    "go away",
    "stop bothering everyone",
    "you are embarrassing",
    "you are not good enough",
    "you are such a loser",
    "nobody wants you here",
    "you are stupid and useless",
    "you are dumb and useless",
    "you are a bad person",
    "you are pathetic",
    "you are annoying everyone"
]

# ==========================================================
# 2. CREATE DATASET
# ==========================================================

messages = (
    normal_messages
    + spam_messages
    + cyberbullying_messages
)

labels = (
    ["Normal"] * len(normal_messages)
    + ["Spam"] * len(spam_messages)
    + ["Cyberbullying"] * len(cyberbullying_messages)
)

# ==========================================================
# 3. TF-IDF
# ==========================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(messages)

# ==========================================================
# 4. LOGISTIC REGRESSION MODEL
# ==========================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X, labels)

# ==========================================================
# 5. HOME PAGE
# ==========================================================

@app.route("/", methods=["GET", "POST"])
def home():

    username = ""
    message = ""

    prediction = None
    confidence = None
    probabilities = None

    if request.method == "POST":

        # Get username entered by user
        username = request.form.get("username", "").strip()

        # Get message entered by user
        message = request.form.get("message", "").strip()

        # Check that message is not empty
        if message:

            # Convert message into TF-IDF
            message_vector = vectorizer.transform([message])

            # Predict message category
            prediction = model.predict(message_vector)[0]

            # Get probability of each category
            probability_values = model.predict_proba(
                message_vector
            )[0]

            # Highest probability = confidence
            confidence = round(
                max(probability_values) * 100,
                2
            )

            # Store all probabilities
            probabilities = {}

            for class_name, probability in zip(
                model.classes_,
                probability_values
            ):
                probabilities[class_name] = round(
                    probability * 100,
                    2
                )
                now = datetime.now()
                
            detection=Detection(
                 username=username
                 if username
                else "Anonymous",
                 message=message,
                 prediction=prediction,
                 confidence=confidence,
                 detection_date=now.strftime("%Y-%m-%d %H:%M:%S"),
                 
            )
            db.session.add(detection)
            db.session.commit()
            
            
                        
               
    # Send values to HTML page
    return render_template(
        "index.html",
        username=username,
        message=message,
        prediction=prediction,
        confidence=confidence,
        probabilities=probabilities
    )
@app.route("/database")
def database():
    records = Detection.query.order_by(
        Detection.id.desc()
    ).all()
    return render_template(
        "database.html",
        records=records
    )

# ==========================================================
# 6. RUN FLASK
# ==========================================================

if __name__ == "__main__":
    app.run(debug=True)