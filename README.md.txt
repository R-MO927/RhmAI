# 🎥 Movie Recommendation System using AI & ML

This project is a **practical application of Artificial Intelligence (AI)** and **Machine Learning (ML)** concepts, focusing on building a **Movie Recommendation System**.  

The system implements **three types of recommendation approaches**: **Collaborative Filtering, Content-Based Filtering, and a Hybrid Model**, each tested on well-known movie datasets.

---

##  Project Overview

The main objective of this project is to **analyze movie datasets**, **build recommendation systems** using multiple approaches, and **evaluate their effectiveness** in providing accurate movie suggestions to users.

---

## 🔹 Implemented Recommendation Types

### 1️⃣ Collaborative Filtering (SVD)
- **Dataset**: [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)
- **Steps**:
  1. Cleaned and merged multiple files (users, movies, and ratings) into a single dataframe called `merged`.
  2. Applied **Singular Value Decomposition (SVD)** to capture **user-item interactions**.
  3. Generated movie recommendations **based on user preferences**.

---

### 2️⃣ Content-Based Filtering
- **Dataset**: [TMDB 5000 Movies Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- **Steps**:
  1. Merged the two main files into one dataframe `df`.
  2. Extracted key features such as:
     - **Genres**
     - **Keywords**
     - **Cast & Crew**
  3. Computed **Cosine Similarity** on movie features to **recommend similar movies** to a selected one.

---

### 3️ Hybrid Recommendation System
- Combined **Collaborative Filtering** and **Content-Based Filtering**:
  - Personalized recommendations based on **user history**.
  - Similar movie suggestions using **content features**.
- Provides **more accurate and diverse recommendations** compared to single methods.

---

##  Notes & Insights
- **Data Cleaning & Preprocessing**:
  - Essential before applying each recommendation technique.
  - Different datasets were prepared according to the method used.
- **Comparison of Models**:
  - Each model's **strengths and weaknesses** were evaluated.
  - **Hybrid models** often outperform individual methods in personalization.

---

##  Example Outputs
- **Content-Based**: If a user likes *Inception*, the system suggests *Interstellar* and *The Prestige*.  
- **Collaborative Filtering**: Suggests movies based on **similar user preferences**.  
- **Hybrid System**: Provides a **blend of both** approaches for higher accuracy.

---

##  Tech Stack & Tools
- **Languages & Libraries**:
  - Python, Pandas, NumPy, Scikit-learn, Surprise
- **Visualization**:
  - Matplotlib, Seaborn
- **Development & Deployment**:
  - Jupyter Notebook, Kaggle






 *Feel free to explore the notebook, try the models, and give feedback!*  
