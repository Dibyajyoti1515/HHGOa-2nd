from retrieval.factory import retrieve


QUESTIONS = [
    # Science
    "What causes earthquakes?",
    "Why do tectonic plates move?",
    "What causes volcanoes to erupt?",
    "How does photosynthesis work?",
    "Why is the sky blue?",
    "What causes a rainbow?",
    "How does the water cycle work?",
    "What causes tides in the ocean?",
    "Why do we have seasons?",
    "What causes lightning?",

    # Biology
    "How does the human heart work?",
    "How does the human digestive system work?",
    "What is the function of the liver?",
    "What does the immune system do?",
    "What is the difference between bacteria and viruses?",
    "How do vaccines work?",
    "Why do humans need sleep?",
    "What causes muscle soreness after exercise?",
    "How does the human brain process information?",
    "What is the role of DNA in the human body?",

    # Earth and environment
    "What causes global warming?",
    "What is the greenhouse effect?",
    "How does deforestation affect biodiversity?",
    "What causes soil erosion?",
    "How does recycling reduce pollution?",
    "What causes air pollution?",
    "What causes ocean pollution?",
    "How does climate change affect the environment?",
    "What causes floods?",
    "What causes droughts?",

    # Physics
    "What is gravity?",
    "How does friction work?",
    "What is the difference between mass and weight?",
    "How does electricity flow through a circuit?",
    "What is the difference between speed and velocity?",
    "How does a lever make work easier?",
    "What causes objects to fall toward Earth?",
    "How does a magnet attract metal?",
    "What is kinetic energy?",
    "What is potential energy?",

    # Technology
    "What is the difference between a CPU and a GPU?",
    "What is RAM used for?",
    "How does an operating system work?",
    "What is the purpose of a database?",
    "What is the difference between SQL and NoSQL?",
    "What is an API?",
    "How does encryption protect data?",
    "What is the difference between HTTP and HTTPS?",
    "What does a DNS server do?",
    "How does cloud computing work?",

    # Programming
    "What is the difference between a compiler and an interpreter?",
    "What is object-oriented programming?",
    "What is functional programming?",
    "How does version control help developers?",
    "What is the difference between a list and a tuple in Python?",
    "How does a neural network work?",
    "What is machine learning?",
    "What is the difference between supervised and unsupervised learning?",
    "How can I improve Python performance?",
    "What is recursion in programming?",

    # History
    "What caused World War One?",
    "What caused World War Two?",
    "How did the Roman Empire fall?",
    "When did the Industrial Revolution begin?",
    "Why was the Berlin Wall built?",
    "What caused the Berlin Wall to fall?",
    "How did the printing press change society?",
    "Why was the Silk Road important?",
    "How were the Egyptian pyramids built?",
    "What caused the French Revolution?",

    # Geography
    "Where is the Amazon rainforest?",
    "What is the largest desert in the world?",
    "How deep is the Mariana Trench?",
    "What is the difference between a country and a continent?",
    "How are mountains formed?",
    "What causes ocean currents?",
    "Where is the Sahara Desert?",
    "What is the longest river in the world?",
    "Why do different countries have different climates?",
    "How are islands formed?",

    # Business and finance
    "What is inflation?",
    "How does inflation affect purchasing power?",
    "What is compound interest?",
    "What is the difference between revenue and profit?",
    "What is a balance sheet?",
    "How does supply and demand affect prices?",
    "What is a monopoly?",
    "What is an oligopoly?",
    "What causes a stock market crash?",
    "How do mutual funds work?",

    # Education and everyday knowledge
    "What is the difference between a bachelor's and master's degree?",
    "How does online learning work?",
    "What is the Montessori method?",
    "How do standardized tests measure learning?",
    "What is the difference between mean and median?",
    "How do you calculate the area of a circle?",
    "What is the Pythagorean theorem?",
    "How does probability work?",
    "What is the difference between a fraction and a decimal?",
    "How does GPS determine your location?",
]


THRESHOLD = 0.90


def evaluate_question(question: str, question_id: int) -> dict:
    """
    Run one question through the same retrieval pipeline
    used by the voice assistant.
    """

    try:
        result = retrieve(
            question,
            mode="hybrid_weighted",
        )

        results = result.get("results", [])

        if not results:
            return {
                "id": question_id,
                "question": question,
                "score": 0.0,
                "passed": False,
                "retrieved_text": "",
                "error": "No retrieval results",
            }

        top_result = results[0]

        score = float(top_result.score)

        payload = top_result.payload or {}

        retrieved_text = (
            payload.get("text_en")
            or payload.get("text")
            or ""
        )

        return {
            "id": question_id,
            "question": question,
            "score": score,
            "passed": score > THRESHOLD,
            "retrieved_text": retrieved_text,
            "error": None,
        }

    except Exception as exc:
        return {
            "id": question_id,
            "question": question,
            "score": 0.0,
            "passed": False,
            "retrieved_text": "",
            "error": str(exc),
        }


def main():

    total_questions = len(QUESTIONS)

    print("=" * 90)
    print("VOICE QUESTION RETRIEVAL EVALUATION")
    print("=" * 90)

    print(f"Total questions : {total_questions}")
    print(f"Threshold       : > {THRESHOLD:.2f}")
    print("Mode            : hybrid_weighted")
    print()

    all_results = []
    passed = []
    failed = []

    # ---------------------------------------------------------
    # Run all questions
    # ---------------------------------------------------------

    for i, question in enumerate(QUESTIONS, start=1):

        row = evaluate_question(
            question=question,
            question_id=i,
        )

        all_results.append(row)

        if row["passed"]:
            passed.append(row)
        else:
            failed.append(row)

        status = "PASS" if row["passed"] else "FAIL"

        print(
            f"[{i:03d}/{total_questions}] "
            f"{status:<4} "
            f"score={row['score']:.4f} "
            f"| {question}"
        )

    # ---------------------------------------------------------
    # Sort best questions
    # ---------------------------------------------------------

    passed.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    failed.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    total = len(all_results)
    passed_count = len(passed)
    failed_count = len(failed)

    pass_percentage = (
        (passed_count / total) * 100
        if total
        else 0
    )

    print("\n")
    print("=" * 90)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 90)

    print(f"Total questions : {total}")
    print(f"Score > 0.90    : {passed_count}")
    print(f"Score <= 0.90   : {failed_count}")
    print(f"Pass rate       : {pass_percentage:.2f}%")

    # ---------------------------------------------------------
    # BEST QUESTIONS FOR VIDEO
    # ---------------------------------------------------------

    print("\n")
    print("=" * 90)
    print("BEST QUESTIONS FOR VIDEO RECORDING")
    print("=" * 90)

    if not passed:
        print("\nNo questions achieved a score above 0.90.")
    else:

        for rank, row in enumerate(passed, start=1):

            print(
                f"\n#{rank}"
            )

            print(
                f"Score      : {row['score']:.4f}"
            )

            print(
                f"Question   : {row['question']}"
            )

            print(
                f"Retrieved  : "
                f"{row['retrieved_text'][:300]}"
            )

    # ---------------------------------------------------------
    # TOP 10 ONLY
    # ---------------------------------------------------------

    print("\n")
    print("=" * 90)
    print("TOP 10 QUESTIONS TO USE IN THE VIDEO")
    print("=" * 90)

    top_10 = passed[:10]

    if not top_10:
        print("No questions above threshold.")
    else:

        for rank, row in enumerate(top_10, start=1):

            print(
                f"{rank:02d}. "
                f"score={row['score']:.4f} "
                f"| {row['question']}"
            )

    # ---------------------------------------------------------
    # FAILED QUESTIONS
    # ---------------------------------------------------------

    print("\n")
    print("=" * 90)
    print("QUESTIONS BELOW OR EQUAL TO 0.90")
    print("=" * 90)

    if not failed:
        print("All questions passed!")
    else:

        for row in failed:

            print(
                f"score={row['score']:.4f} "
                f"| {row['question']}"
            )

    # ---------------------------------------------------------
    # Retrieval statistics
    # ---------------------------------------------------------

    scores = [
        row["score"]
        for row in all_results
        if row["error"] is None
    ]

    if scores:

        average_score = sum(scores) / len(scores)
        best_score = max(scores)
        worst_score = min(scores)

        print("\n")
        print("=" * 90)
        print("SCORE STATISTICS")
        print("=" * 90)

        print(f"Average score : {average_score:.4f}")
        print(f"Best score    : {best_score:.4f}")
        print(f"Worst score   : {worst_score:.4f}")


if __name__ == "__main__":
    main()