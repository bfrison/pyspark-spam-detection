## Instructions:
 - To create conda environments, execute:
   `conda env create -f requirements.txt -n spam-detection`
 - To activate environment, execute:  
   `conda activate spam-detection`
 - To fit model, execute:  
   `spark-submit pyspark_model.py`
 - Make a copy of `.env\_sample` called `.env` and fill in your Twilio SID and token
 - To retrieve your Twilio messages as a spark DataFrame and classify them as ham or spam, first execute:  
   `pyspark`  
   to open a pyspark console, then:
```
    import script
    df_sms = script.get_sms_and_predict(sc)
```
