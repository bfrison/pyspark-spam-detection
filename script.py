import os
import warnings

from dotenv import load_dotenv
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import Tokenizer, Word2VecModel

from twilio.rest import Client

load_dotenv()  # load environment variables

TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
twilio_api = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

warnings.filterwarnings("ignore")


def get_sms_and_predict(sc):
    t = Tokenizer.load('models/tokenizer')
    t.setInputCol('body')
    word2Vec_model = Word2VecModel.load('models/word2Vec_model')
    rf_model = RandomForestClassificationModel.load('models/rf_model')

    df_sms = (
        sc.parallelize(twilio_api.messages.stream())
        .map(lambda x: (x.sid, x.body))
        .toDF(schema=['sid', 'body'])
    )
    df_sms = t.transform(df_sms)
    df_sms = word2Vec_model.transform(df_sms)
    df_sms = rf_model.transform(df_sms)

    return df_sms
