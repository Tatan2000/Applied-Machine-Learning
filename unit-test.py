import os, time, yaml, pickle, joblib, subprocess, requests, signal
from src.score import score
import numpy as np


##########################################################################################################################



config_file_path = os.path.join('config', 'test_config', 'unit.yaml')

with open(config_file_path) as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)



# extract the input texts
spam_text = config['input_text']['spam']
ham_text = config['input_text']['ham']

tfidf = pickle.load(open(config['tfidf_save_path'], 'rb'))
best_clf = joblib.load(config['best_clf_save_path'])
default_threshold = config['threshold']

is_spam, proba = score(
    input_text=spam_text,
    vectorizer=tfidf,
    classifier=best_clf,
    threshold=default_threshold
)


class TestClass:

    def test_smoke(self):
        assert (is_spam != None)
        assert (proba != None)
    
    def test_format(self):
        assert type(spam_text) == str
        assert type(default_threshold) == float
        assert type(is_spam) == bool
        assert type(proba) == np.float64

    def test_pred_value(self):
        assert (is_spam == True or is_spam == False)
   
    def test_prop_value(self):
        assert ((proba >= 0) and (proba <= 1))

    def test_pred_th0(self):
        label, _ = score(
            input_text=spam_text,
            vectorizer=tfidf,
            classifier=best_clf,
            threshold=0
        )
        assert (label == True)

    def test_pred_th1(self):
        label, _ = score(
            input_text=spam_text,
            vectorizer=tfidf,
            classifier=best_clf,
            threshold=1
        )
        assert (label == False)
    
    def test_spam(self):
        label, _ = score(
            input_text=spam_text,
            vectorizer=tfidf,
            classifier=best_clf,
            threshold=default_threshold
        )
        assert (label == True)

    def test_ham(self):
        label, _ = score(
            input_text=ham_text,
            vectorizer=tfidf,
            classifier=best_clf,
            threshold=default_threshold
        )
        assert (label == False)
    
    # testing the default spam message in docker container
    def test_docker(self):
        subprocess.run(['docker', 'build', '-t', 'spam_classifier', '.'], check=True)

        container = subprocess.Popen(['docker', 'run', '-p', '5000:5000', '-d', 'spam_classifier'], stdout=subprocess.PIPE)

        time.sleep(5)

        try:
           
            spam_data = {'sms_text': spam_text}
            
           
            response_post = requests.post('http://localhost:5000/score', data=spam_data)

            
            assert (response_post.status_code == 200)

            
            output_json_str = response_post.data.decode()
            assert "prediction" in output_json_str
            assert "propensity" in output_json_str
        
        finally:
            os.kill(container.pid, signal.SIGTERM)
            container.wait()


##########################################################################################################################


# run the following command in terminal to produce coverage report :
# coverage run -m pytest test/unit_test.py && coverage report -m > coverage_reports/unit_test_coverage.txt
