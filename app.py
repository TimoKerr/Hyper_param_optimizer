from mylib import util
from mylib import hyperoptimize


if __name__ == "__main__":
    df = util.read_data("/home/timoose/MLprojects/Hyper_param_optimizer/data/train.csv")

    train_x = df.drop("price_range", axis=1).values
    train_y = df.price_range.values

    GridSearchModel = hyperoptimize.GridSearch()
    GridSearchModel.fit(train_x, train_y)
    print(GridSearchModel.best_score_)
    print(GridSearchModel.best_estimator_.get_params())

    RandomSearchModel = hyperoptimize.RandomSearch()
    RandomSearchModel.fit(train_x, train_y)
    print(RandomSearchModel.best_score_)
    print(RandomSearchModel.best_estimator_.get_params())

    SkoptSearchResult = hyperoptimize.SkoptSearch(train_x, train_y)
    print(SkoptSearchResult)

    HyperoptSearchResult = hyperoptimize.HyperOptSearch(train_x, train_y)
    print(HyperoptSearchResult)

    OptunaSearchResult = hyperoptimize.OptunaSearch(train_x, train_y)
