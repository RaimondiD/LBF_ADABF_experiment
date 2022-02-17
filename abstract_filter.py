import serialize

class Abstract_Filter:
    def query(self, query_set): pass
     
    def save(self, path):
        serialize.save_model(self,path) 