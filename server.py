import random
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from flask import Flask, jsonify, request
from sqlalchemy import create_engine, Column, String, Integer, UniqueConstraint
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base
import numpy as np
from sqlalchemy.sql.expression import func
from flask_cors import CORS
import tensorflow as tf
import tensorflow_hub as hub
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

Base = declarative_base()

nltk.download('stopwords')
nltk.download('punkt')
def preprocess_text(text):

    text = text.lower()


    text = text.translate(str.maketrans("", "", string.punctuation))


    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(word) for word in filtered_text]
    preprocessed_text = " ".join(lemmatized_text)

    return preprocessed_text
def euclidean_distance(vector1, vector2):
    return np.linalg.norm(np.array(vector1) - np.array(vector2))

class Storyes(Base):
    __tablename__ = 'Storyes'
    __table_args__ = (
        UniqueConstraint('storyid', name='unique_id'),
    )

    storyid = Column(Integer, primary_key=True, autoincrement=True)
    storyText = Column(String, nullable=False)
    storyVector = Column(String, nullable=False)

app = Flask(__name__)
CORS(app)

@app.route('/api/add_story', methods=['POST'])
def add_story():
    data = request.get_json()
    story_text = data.get("storyText")
    story_text =preprocess_text(story_text)
    story_text = [story_text]
    story_vector = embed(story_text)
    new_story = Storyes(storyText=story_text, storyVector=story_vector.numpy().tolist())
    session.add(new_story)
    session.commit()
    new_story_id = new_story.storyid
    data = {
        "storyId": new_story_id
    }
    try:
        response = jsonify(data)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_stories', methods=['GET'])
def get_stories():
    id_value = request.cookies.get('id')
    q_value = request.args.get('q')

    if id_value is not None:
        id_value = int(id_value)
        story_vector = session.query(Storyes.storyVector).filter(Storyes.storyid == id_value).first()
        story_vector = np.array([float(value) for value in story_vector[0].strip('{}').split(',')])
        topics = ["Stress and Coping Strategies", "Understanding and Overcoming Depression",
                  "Anxiety and Coping Mechanisms", "Self-Esteem and Building Confidence"
            , "Interpersonal Relationships and Communication Challenges", "Emotional Management in Daily Life",
                  "Adapting to Change and Rethinking Life Goals", "Self-Understanding and Personal Growth"
            , "Sleep Issues and Solutions", "Addictions and Strategies for Recovery"]
        dataFromDatabase = session.query(Storyes.storyid, Storyes.storyVector).filter(Storyes.storyid != id_value).all()
        result_with_distances = []
        for row in dataFromDatabase:
            story_id = row[0]
            vector_from_database = [float(value) for value in row[1].strip('{}').split(',')]
            distance = tf.keras.losses.cosine_similarity(tf.cast(story_vector, dtype=tf.double),  tf.cast(vector_from_database, dtype=tf.double))



            result_with_distances.append((story_id, row[1], distance))
        result_with_distances.sort(key=lambda x: x[2])
        distances = []
        if q_value is None:
         distances = result_with_distances[:5]
        else:
         distances = result_with_distances[:int(q_value)]
        top_ids = [item[0] for item in distances]
        max_index = np.argmax(story_vector)
        topicForClient  = topics[random.randint(0, 9)]
        result_texts = session.query(Storyes.storyText).filter(Storyes.storyid.in_(top_ids)).all()
        result_texts = [result[0] for result in result_texts]
        data = {
           "textsForClient": result_texts,
           "mainTopic": topicForClient
        }
        return jsonify(data)
    else:
        random_texts = []
        if q_value is None:
            random_texts = session.query(Storyes.storyText).order_by(func.random()).limit(q_value).all()
        else:
            random_texts = session.query(Storyes.storyText).order_by(func.random()).limit(5).all()
        random_texts = [text[0] for text in random_texts]
        data = {
            "textsForClient": random_texts
        }
        return jsonify(data)

if __name__ == '__main__':
    postgresql_url = 'postgresql://postgres:12345@localhost:5432/psycho'
    engine = create_engine(postgresql_url)

    # Создаем объект сессии для взаимодействия с базой данных
    Session = sessionmaker(bind=engine)
    session = Session()

    # Запуск Flask-приложения
    app.run(debug=True, port=8000, host="0.0.0.0")