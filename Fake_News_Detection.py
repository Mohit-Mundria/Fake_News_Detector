'''Here we made a fun that will take required inputs from the user and, give those inputs to the Tokenizer to tokenize and then those tokenised input to the Model to predict
that the Given Input is Fake or Real.'''
def Fake_News_Detection():
    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        import joblib
        model=load_model(r"C:\Users\ACER\Documents\Downloads\fake_news_detector_model.h5")
        token=joblib.load(r"C:\Users\ACER\Documents\Downloads\tokenizer_Fake_or_True.pkl")
        title=input("Enter the Title of the news:")
        text=input("Enter the Text or discription of the news:")
        subject=input("Enter the subject of the news from (Worldnews, politics, left-news): ")
        data=title+" "+text + " "+subject
        sequences = token.texts_to_sequences(data)
        padded_sequences = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')

        prediction=model.predict(padded_sequences)[0][0]
        threshold=0.5
        if prediction>=threshold:
            print("The Model predict the news as Fake")
        else:
            print("The model predict the news as Real")
    except ValueError as e1:
        print("Please provide valid values(numeric) for the inputs.",e1)
    except Exception as e2:
        print("Something wents wrong in the execution of the code", e2)
    finally:
        print("Thanks for using this Fake_News_Detection model.")

if __name__=="__main__":
    Fake_News_Detection()
    
