from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = "static"

def run(team_names, svm):
    @app.route("/")
    def hello():
        # returning string
        return render_template('index.html', len = len(team_names), team_names = team_names)

    @app.route("/test", methods=['GET', 'POST'])
    def test():
        team_1 = request.form.get('team_1')
        team_2 = request.form.get('team_2')
        print(str(team_1), str(team_2))

        prob1, text1 = svm.prediction(team_1, team_2)

        return (text1)  # just to see what select is

    app.run(debug = True)