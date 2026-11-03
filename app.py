from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictionPipeline

app = Flask(__name__, template_folder="templates")


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():

    if request.method == "GET":
        return render_template("index.html")

    try:
        # Collect form data
        data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("race_ethnicity"),
            parental_level_of_education=request.form.get(
                "parental_level_of_education"
            ),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get(
                "test_preparation_course"
            ),
            reading_score=float(request.form.get("reading_score")),
            writing_score=float(request.form.get("writing_score"))
        )

        # Convert input into DataFrame
        pred_df = data.get_data_as_data_frame()

        # Prediction
        predict_pipeline = PredictionPipeline()
        results = predict_pipeline.predict(pred_df)

        prediction = round(float(results[0]), 2)

        return render_template(
            "index.html",
            results=prediction
        )

    except Exception as e:
        app.logger.exception("Prediction failed")

        return render_template(
            "index.html",
            error="Unable to generate prediction. Please check your inputs and try again."
        )


if __name__ == "__main__":
    app.run(debug=True)