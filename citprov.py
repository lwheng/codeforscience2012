import Utils

class citprov:
  def __init__(self):
    self.pickler = Utils.pickler()
    # Load model for prediction
    #self.model = self.pickler.loadPickle(self.pickler.pathModel)

  def predict(self, model, query):
    # Process query (Might have multiple steps)
    queryProcessed = ""
    # Predict query using model
    #return model.predict(queryProcessed)
    return 'Hello World!'
