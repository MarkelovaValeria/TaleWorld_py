import ast
import gensim
import pandas as pd
from LDAFastTextClass import LDAFastTextModel
from ABSAClass import ABSAModel
from config import config
from TextCleanerClass import TextCleaner

def main(df):
    cleaner = TextCleaner()
    df = cleaner.clean_dataframe(df)
    
    df["token"] = df["tokens"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    bigram = gensim.models.Phrases(df["token"], min_count=5, threshold=10)
    trigram = gensim.models.Phrases(bigram[df["token"]], threshold=10)

    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    df["token"] = df["token"].apply(lambda x: trigram_mod[bigram_mod[x]])

    aspects = ["food", "service", "place"]
    lda_ft_model = LDAFastTextModel(
        df=df,
        num_topics=3,
        vector_size=100,
        aspects=aspects,
        workers=1
    )

    lda_model, fasttext_model, df_sim, df_with_features = lda_ft_model.run()

    print("Таблиця схожості тем і аспектів:")
    print(df_sim)

    print("Оновлений DataFrame з FastText ознаками:")
    print(df_with_features.head())


    test_phrases = [

        # ---------------- POSITIVE ----------------
        ("I love this food", "Positive"),

        ("The food was excellent and although the service was a bit slow, "
        "the staff were friendly and the atmosphere made up for everything.", "Positive"),

        ("Amazing desserts and the main dishes were well-seasoned. "
        "Even though it was crowded, the place felt cozy and the waiters were polite.",
        "Positive"),

        ("I enjoyed the fresh ingredients and the creative menu. "
        "Service wasn’t perfect, but overall the experience was great.",
        "Positive"),

        ("Great value for money! The burgers were juicy, the drinks were well prepared, "
        "and the interior design was beautiful.", 
        "Positive"),

        ("Our table was delayed, but once seated, the rest of the evening was flawless. "
        "Food, music, and staff — all excellent.",
        "Positive"),

        # ---------------- NEGATIVE ----------------
        ("I hate this food", "Positive"),
        ("The pasta tasted old, the place smelled weird, and even though the waiter tried, "
        "the overall experience was disappointing.", 
        "Negative"),

        ("Terrible service. The food itself wasn't the worst, but waiting 50 minutes "
        "for a simple order ruined everything.", 
        "Negative"),

        ("The restaurant looks nice from outside, but the food was bland and overpriced, "
        "and the staff acted as if we were bothering them.",
        "Negative"),

        ("A really uncomfortable experience — loud music, dirty tables, and the steak was "
        "so tough I could barely cut it.", 
        "Negative"),

        ("Portions were tiny, the soup was cold, and even though the waiter apologized, "
        "the whole thing felt like a waste of money.",
        "Negative"),

        # ---------------- MIXED / HARD CASES ----------------
        ("The pizza was delicious, but everything else was disappointing: "
        "rude staff, dirty floor, and extremely long waiting time.", 
        "Negative"),

        ("Service was chaotic and disorganized, but honestly the food was some of the best "
        "I’ve ever tasted.", 
        "Positive"),

        ("Great location and beautiful interior, but the meals were just average "
        "and definitely not worth the price.", 
        "Negative"),

        ("The appetizers were incredible, but the main dishes were dry and flavorless. "
        "Still, the waiter did their best to fix things.",
        "Negative"),

        ("Fantastic cocktails and friendly bartenders, but the food was mediocre "
        "and arrived cold. Nice place though.",
        "Positive"),
    ]




    absa_model = ABSAModel(
        df=df_with_features,      
        fasttext_model=fasttext_model,
        aspects=aspects,
        text_column="Review_corrected"
    )

    absa_model.fit()

    final_results = absa_model.evaluate_restaurants()

    print("=== SENTIMENT TEST RESULTS ===\n")

    for text, expected in test_phrases:
        print(f"TEXT: {text}\nEXPECTED SENTIMENT: {expected}")
        aspect_scores = absa_model.analyze_review_aspects_score(text)
        
        for aspect, vals in aspect_scores.items():
            print(f"{aspect:<8} - {vals['combined']:.3f}")
        print("-" * 80)

if __name__ == "__main__":
    df = pd.read_csv("Restaurant_reviews.csv")
    main(df)
