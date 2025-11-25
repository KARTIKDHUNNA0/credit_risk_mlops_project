import pandera as pa

schema = pa.DataFrameSchema({
    "SK_ID_CURR": pa.Column(int, nullable=False),
    "TARGET": pa.Column(int, nullable=False),
})

def validate_application_train(df):
    return schema.validate(df)
