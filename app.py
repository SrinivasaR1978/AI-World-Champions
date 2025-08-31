import os
import uuid
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Save plots without GUI
import matplotlib.pyplot as plt
from flask import send_from_directory
from flask import Flask, request, render_template, redirect, url_for, session, flash
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from mpl_toolkits.mplot3d import Axes3D

app = Flask(__name__)

# Example AI topics (later you can fetch from DB)
topics = [
    {
        "title": "Supervised Learning",
        "category": "ml",
        "category_name": "Machine Learning",
        "difficulty": "beginner",
        "description": "Learning with labeled training data to make predictions on new data.",
        "details": {
            "structure": {
                "Regression": ["Linear Regression", "Polynomial Regression", "Ridge/Lasso Regression"],
                "Classification": ["Logistic Regression", "Decision Trees", "Random Forest", "SVM"]
            },
            "frameworks": ["Scikit-learn", "TensorFlow", "PyTorch", "XGBoost", "LightGBM", "CatBoost"],
            "applications": ["Email spam detection", "Credit scoring", "Medical diagnosis", "Image classification"],
            "algorithms": ["Linear Regression", "Decision Trees", "Random Forest", "Support Vector Machines"]
        }
    },
    {
        "title": "Unsupervised Learning",
        "category": "ml",
        "category_name": "Machine Learning",
        "difficulty": "intermediate",
        "description": "Finding hidden patterns in data without labeled examples.",
        "details": {
            "structure": {
                "Clustering": ["K-Means", "Hierarchical Clustering", "DBSCAN"],
                "Dimensionality Reduction": ["PCA", "t-SNE", "Autoencoders"]
            },
            "frameworks": ["Scikit-learn", "TensorFlow", "PyTorch"],
            "applications": ["Customer segmentation", "Anomaly detection", "Market basket analysis"],
            "algorithms": ["K-Means", "PCA", "t-SNE", "DBSCAN"]
        }
    },
    {
        "title": "Reinforcement Learning",
        "category": "rl",
        "category_name": "Reinforcement Learning",
        "difficulty": "advanced",
        "description": "Learning through interaction with environment using rewards and penalties.",
        "details": {
            "structure": {
                "Policy-based": ["Policy Gradients", "Actor-Critic"],
                "Value-based": ["Q-Learning", "Deep Q-Networks (DQN)"]
            },
            "frameworks": ["Stable Baselines3", "TensorFlow", "PyTorch"],
            "applications": ["Robotics", "Game AI", "Autonomous driving"],
            "algorithms": ["Q-Learning", "Policy Gradients", "Actor-Critic", "DQN"]
        }
    },
    {
        "title": "Deep Learning",
        "category": "dl",
        "category_name": "Deep Learning",
        "difficulty": "intermediate",
        "description": "Neural networks with multiple layers for representation learning.",
        "details": {
            "structure": {
                "Architectures": ["CNN", "RNN", "GAN", "Transformer"],
                "Training": ["Backpropagation", "Batch Normalization", "Dropout"]
            },
            "frameworks": ["TensorFlow", "Keras", "PyTorch"],
            "applications": ["Image recognition", "Speech recognition", "Generative art"],
            "algorithms": ["Convolutional Neural Networks", "Recurrent Neural Networks", "GANs", "Transformers"]
        }
    },
    {
        "title": "Natural Language Processing (NLP)",
        "category": "nlp",
        "category_name": "NLP",
        "difficulty": "intermediate",
        "description": "Teaching machines to understand, interpret and generate human language.",
        "details": {
            "structure": {
                "Classical NLP": ["Bag of Words", "TF-IDF", "Word2Vec"],
                "Modern NLP": ["Transformers", "BERT", "GPT"]
            },
            "frameworks": ["NLTK", "spaCy", "HuggingFace Transformers"],
            "applications": ["Chatbots", "Machine Translation", "Sentiment Analysis", "Text Summarization"],
            "algorithms": ["Naive Bayes", "RNNs", "LSTMs", "Transformers"]
        }
    }
]
categories = {
    "AI Assistants": {
        "desc": "Conversational AI that can help with various tasks, answer questions, and assist with problem-solving.",
        "tools": ["ChatGPT", "Grok", "Claude", "+1 more"]
    },
    "Video Generation": {
        "desc": "AI-powered tools for creating, editing, and enhancing video content automatically.",
        "tools": ["Synthesia", "Google Veo", "OpusClip"]
    },
    "Image Generation": {
        "desc": "Create stunning images, artwork, and visual content using AI-powered generation tools.",
        "tools": ["GPT-4o", "Midjourney"]
    },
    "Meeting Assistants": {
        "desc": "AI tools that help record, transcribe, and summarize meetings for better productivity.",
        "tools": ["Fathom", "Nyota"]
    },
    "Automation": {
        "desc": "Workflow automation tools that use AI to streamline repetitive tasks and processes.",
        "tools": ["n8n", "Manus"]
    },
    "Research": {
        "desc": "AI-enhanced research tools for gathering, analyzing, and synthesizing information.",
        "tools": ["Deep Research", "NotebookLM"]
    },
    "Writing": {
        "desc": "AI writing assistants for content creation, editing, and creative writing.",
        "tools": ["Rytr", "Sudowrite"]
    },
    "Search Engines": {
        "desc": "Next-generation search engines powered by AI for more intelligent information retrieval.",
        "tools": ["Google AI Mode", "Perplexity", "ChatGPT Search"]
    },
    "Graphic Design": {
        "desc": "AI-powered design tools for creating professional graphics, logos, and visual content.",
        "tools": ["Canva Magic Studio", "Looka"]
    },
    "App Builders & Coding": {
        "desc": "AI-powered development tools for building applications and writing code more efficiently.",
        "tools": ["Lovable", "Cursor"]
    },
    "Knowledge Management": {
        "desc": "AI tools for organizing, searching, and managing knowledge and information.",
        "tools": ["Notion Q&A", "Guru"]
    },
    "Email": {
        "desc": "AI-enhanced email tools for writing, managing, and optimizing email communication.",
        "tools": ["HubSpot Email Writer", "Fyxer", "Shortwave"]
    },
    "Scheduling": {
        "desc": "AI scheduling assistants that optimize time management and calendar coordination.",
        "tools": ["Reclaim", "Clockwise"]
    },
    "Presentations": {
        "desc": "AI tools for creating compelling presentations and slide decks automatically.",
        "tools": ["Gamma", "Copilot for PowerPoint"]
    },
    "Resume Builders": {
        "desc": "AI-powered resume and CV builders for creating professional job applications.",
        "tools": ["Teal", "Kickresume"]
    },
    "Voice Generation": {
        "desc": "AI voice synthesis and generation tools for creating realistic speech and audio.",
        "tools": ["ElevenLabs", "Murf"]
    },
    "Music Generation": {
        "desc": "AI-powered music creation tools for composing songs, beats, and audio content.",
        "tools": ["Suno", "Udio"]
    },
    "Marketing": {
        "desc": "AI marketing tools for creating campaigns, ads, and promotional content.",
        "tools": ["AdCreative", "AirOps"]
    },
}
app.secret_key = "secret123"   # Needed for sessions

# Dummy user database (use real DB in production)
users = {"admin": "password123", "hr": "hrpass"}


# @app.route("/")
# def login():
#     return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username in users and users[username] == password:
            session["user"] = username
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password", "danger")
            return redirect(url_for("login"))
    return render_template("login.html")


@app.route("/home")
def home():
    if "user" in session:
        return render_template("home.html", user=session["user"])
    else:
        return redirect(url_for("login"))


@app.route("/Signout")
def logout():
    session.pop("user", None)
    flash("You have been Signed out.", "info")
    return redirect(url_for("login"))


@app.route('/ai',methods=["GET"])
def ai_page():
    search = request.args.get("search", "").lower()
    category = request.args.get("category", "")
    difficulty = request.args.get("difficulty", "all")

    results = topics

    # Apply filters
    if search:
        results = [t for t in results if search in t["title"].lower()]
    if category:
        results = [t for t in results if t["category"] == category]
    if difficulty != "all":
        results = [t for t in results if t["difficulty"] == difficulty]

    return render_template("ai.html", results=results, request=request)

@app.route('/aitools')
def aitool_page():
    return render_template('aitools.html', categories=categories)


@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/Documents')
def Documents():
    return render_template('Documents.html')

app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Store dataframes in memory
dataframes = {}

@app.route("/projects", methods=["GET", "POST"])
def projects():
    if request.method == "POST" and "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return render_template("projects.html", error="❌ No file selected")

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        if file.filename.endswith(".csv"):
            df = pd.read_csv(filepath)
        elif file.filename.endswith(".xlsx"):
            df = pd.read_excel(filepath)
        else:
            return render_template("projects.html", error="❌ Unsupported file format")

        dataframes[file.filename] = df
        rows, cols = df.shape
        columns = list(df.columns)
        preview = df.head().to_html(classes="table", index=False)

        return render_template("projects.html",
                               filename=file.filename,
                               rows=rows, cols=cols,
                               columns=columns,
                               preview=preview,
                               stage="select")

    # Regression
    if request.method == "POST" and "target" in request.form:
        filename = request.form["filename"]
        target = request.form["target"]
        features = request.form.getlist("features")
        chart_type = request.form.get("chart_type")  # NEW

        df = dataframes.get(filename)
        if df is None:
            return render_template("projects.html", error="❌ File not found in memory")

        X, y = df[features], df[target]
        results, plot_filename = {}, None

        # Auto detect regression type
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
            model_type = "linear"
        else:
            model_type = "logistic"

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_type == "linear":
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results["Model"] = "Linear Regression"
            results["MSE"] = round(mean_squared_error(y_test, y_pred), 4)
            results["R²"] = round(r2_score(y_test, y_pred), 4)
            results["Coefficients"] = dict(zip(features, model.coef_))

        else:
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results["Model"] = "Logistic Regression"
            results["Accuracy"] = round(accuracy_score(y_test, y_pred), 4)
            results["Coefficients"] = dict(zip(features, model.coef_[0]))

        # ==========================
        # Extra Plots
        # ==========================
        plt.figure()
        if chart_type == "line":
            df[features + [target]].plot(kind="line")
        elif chart_type == "bar":
            df[features + [target]].head(10).plot(kind="bar")
        elif chart_type == "hist":
            df[features].plot(kind="hist", alpha=0.7)
        elif chart_type == "scatter" and len(features) >= 1:
            plt.scatter(df[features[0]], df[target])
            plt.xlabel(features[0])
            plt.ylabel(target)
        elif chart_type == "pie":
            df[target].value_counts().plot(kind="pie", autopct='%1.1f%%')
        elif chart_type == "3d" and len(features) >= 2:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(df[features[0]], df[features[1]], df[target])
            ax.set_xlabel(features[0])
            ax.set_ylabel(features[1])
            ax.set_zlabel(target)

        plot_filename = f"uploads/plot_{uuid.uuid4().hex}.png"
        plt.savefig(plot_filename)
        plt.close()

        return render_template("projects.html",
                               filename=filename,
                               target=target,
                               features=features,
                               results=results,
                               plot=plot_filename,
                               stage="results")

    return render_template("projects.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
