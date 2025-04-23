from metaflow import FlowSpec, step, kubernetes, resources, retry, timeout, catch, conda_base

@conda_base(python="3.9.16", libraries={"scikit-learn": "1.2.2", "numpy": "1.23.5"})
class ClassifierTrainFlow(FlowSpec):

    @step
    def start(self):
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        import numpy as np

        X, y = datasets.load_wine(return_X_y=True)
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(X, y, test_size=0.2, random_state=0)
        print("Data loaded successfully")
        self.lambdas = np.arange(0.001, 1, 0.01)
        self.next(self.train_lasso, foreach='lambdas')

    @kubernetes
    @resources(cpu=1, memory=4000)
    @retry(times=3)
    @timeout(seconds=300)
    @catch(var='error')
    @step
    def train_lasso(self):
        from sklearn.linear_model import Lasso

        self.model = Lasso(alpha=self.input)
        self.model.fit(self.train_data, self.train_labels)
        self.next(self.choose_model)

    @step
    def choose_model(self, inputs):
        def score(inp):
            return inp.model, inp.model.score(inp.test_data, inp.test_labels)

        self.results = sorted(map(score, inputs), key=lambda x: -x[1])
        self.model = self.results[0][0]
        self.next(self.end)

    @step
    def end(self):
        print('Scores:')
        print('\n'.join('%s %f' % res for res in self.results))
        print('Model:', self.model)

if __name__ == '__main__':
    ClassifierTrainFlow()