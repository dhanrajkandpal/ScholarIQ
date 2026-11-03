from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictionPipeline

app = Flask(__name__, template_folder="templates")


# Home Page
@app.route("/")
def home():
    return render_template("home.html")


# Prediction Page
@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():

    if request.method == "GET":
        return render_template("index.html")

    else:
        try:
            # Get input data from HTML form
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
                reading_score=int(request.form.get("reading_score")),
                writing_score=int(request.form.get("writing_score"))
            )

            # Convert input data into DataFrame
            pred_df = data.get_data_as_data_frame()

            print(pred_df)

            # Initialize prediction pipeline
            predict_pipeline = PredictionPipeline()

            # Generate prediction
            results = predict_pipeline.predict(pred_df)

            return render_template(
                "index.html",
                results=round(float(results[0]), 2)
            )

        except Exception as e:
            return render_template(
                "index.html",
                error=str(e)
            )


if __name__ == "__main__":
    app.run(debug=True)