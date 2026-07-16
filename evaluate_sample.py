import os
import pickle
import sys
import json

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')
ROOT_DIR = os.path.dirname(__file__)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from text_utils import infer_priority_override, normalize_complaint_text

def load_pickle(name):
    path = os.path.join(WEIGHTS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def predict_text(text):
    vectorizer = load_pickle('vectorizer.pkl')
    classifier = load_pickle('classifier.pkl')
    priority_clf = load_pickle('priority_classifier.pkl')

    result = {
        'status': 'Error',
        'match_score': 0.0,
        'category': None,
        'priority': None,
        'expected_days': None,
        'reason': ''
    }

    if vectorizer is None:
        result['reason'] = 'vectorizer missing'
        return result

    normalized_text = normalize_complaint_text(text)
    X = vectorizer.transform([normalized_text])

    if classifier is None:
        result['category'] = 'General'
    else:
        try:
            result['category'] = str(classifier.predict(X)[0])
        except Exception as e:
            result['category'] = 'General'
            result['reason'] = f'classifier error: {e}'

    override_priority = infer_priority_override(text)
    if override_priority is not None:
        result['priority'] = override_priority
    elif priority_clf is not None:
        try:
            raw = str(priority_clf.predict(X)[0])
            result['priority'] = raw.capitalize()
        except Exception as e:
            result['priority'] = 'Medium'
            result['reason'] = result.get('reason','') + f'; priority error: {e}'
    else:
        # fallback heuristic
        result['priority'] = 'High' if len(text) > 200 else ('Medium' if len(text) > 50 else 'Low')

    result['expected_days'] = 3 if result['priority'].lower()=='high' else (5 if result['priority'].lower()=='medium' else 7)
    result['status'] = 'OK'
    result['match_score'] = 1.0
    return result


if __name__ == '__main__':
    if len(sys.argv) > 1:
        text = ' '.join(sys.argv[1:])
    else:
        text = 'measure accident by pothole on road'

    out = predict_text(text)
    print(json.dumps(out, ensure_ascii=False, indent=2))
