import polars as pl
from sklearn.model_selection import train_test_split
from catboost import CatBoostRanker, Pool


def user_features(dataset: pl.DataFrame) -> pl.DataFrame:
    return (
        dataset
        .group_by("user_id")
        .agg(
            (pl.col("action_type") == "AT_CartUpdate").sum().alias("num_cart_updates"),
            (pl.col("action_type") == "AT_View").sum().alias("num_views"),
            pl.col("request_id").len().alias("num_requests"),
            (
                pl.when(pl.col("action_type") == "AT_CartUpdate")
                .then(pl.col("request_id"))
                .otherwise(pl.lit(None))
                .unique().len()
            ).alias("cart_update_requests"),
        )
        .select(
            pl.col("user_id"),
            (pl.col("num_cart_updates") / pl.col("num_views")).alias("ctr"),
            (pl.col("cart_update_requests") / pl.col("num_requests")).alias("conversion_rate"),
        )
    )

def item_features(dataset: pl.DataFrame) -> pl.DataFrame:
    return (
        dataset
        .group_by("product_id")
        .agg(
            (pl.col("action_type") == "AT_CartUpdate").sum().alias("num_cart_updates"),
            (pl.col("action_type") == "AT_View").sum().alias("num_views"),
            pl.col("request_id").len().alias("num_requests"),
            (
                pl.when(pl.col("action_type") == "AT_CartUpdate")
                .then(pl.col("request_id"))
                .otherwise(pl.lit(None))
                .unique().len()
            ).alias("cart_update_requests"),
        )
        .select(
            pl.col("product_id"),
            (pl.col("num_cart_updates") / pl.col("num_views")).alias("ctr"),
            (pl.col("cart_update_requests") / pl.col("num_requests")).alias("conversion_rate"),
        )
    )


def train_cbm_ranker(features_data, model_data, plot=True):
    dataset = (
        model_data
        .sort("timestamp")
        .select(
            pl.col("user_id"),
            pl.col("product_id"),
            pl.col("request_id"),
            pl.when(pl.col("action_type") == "AT_CartUpdate").then(1).otherwise(0).alias("target"),
        )
        .join(user_features(features_data), on="user_id", how="left")
        .join(item_features(features_data), on="product_id", how="left", suffix="_item")
        .filter(pl.col("request_id").is_not_nan())
    )

    x_train, x_test = train_test_split(dataset, test_size=0.2, shuffle=False)
    x_train = x_train.sort("request_id")
    x_test = x_test.sort("request_id")

    train_pool = Pool(
        x_train.drop("target", "user_id", "product_id", "request_id").to_numpy(),
        feature_names=x_train.drop("target", "user_id", "product_id", "request_id").columns,
        group_id=x_train["request_id"].cast(str).to_numpy(),
        label=x_train["target"].to_numpy()
    )
    test_pool = Pool(
        x_test.drop("target", "user_id", "product_id", "request_id").to_numpy(),
        feature_names=x_test.drop("target", "user_id", "product_id", "request_id").columns,
        group_id=x_test["request_id"].cast(str).to_numpy(),
        label=x_test["target"].to_numpy()
    )

    model = CatBoostRanker(eval_metric="NDCG", iterations=100)
    model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=30, plot=plot)
    return model
