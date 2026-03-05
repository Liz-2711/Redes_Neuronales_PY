class Eval:
   
    def __init__(self):
        self.total = 0
        self.correct = 0
        #se guardan los errores 
        self.errors = []
    
    def eval_uno(self, prediction, label, image=None, index=None):
        
        self.total += 1
        
        if prediction == label:
            self.correct += 1
        else:
            self.errors.append({
                "index": index,
                "image": image,
                "prediction": prediction,
                "label": label
            })
    
    def accuracy(self):
        
        if self.total == 0:
            return 0.0
        return self.correct / self.total
    
    def acc_percentage(self):
        #Porcentaje 
        return self.accuracy() * 100
    
    def errors(self, limit=None):
        
        if limit is None:
            return self.errors
        return self.errors[:limit]
    
    def Resultado(self):
      
        return {
            "total": self.total,
            "correct": self.correct,
            "incorrect": self.total - self.correct,
            "accuracy": self.accuracy(),
            "accuracy_percentage": self.acc_percentage()
        }
    
    