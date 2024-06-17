from sqlalchemy.orm import Session
from sqlalchemy import Engine, exc, sql


def check_db_availability(engine: Engine, check_vector: bool = False) -> None:
    try:
        with engine.connect() as conn:
            if check_vector:
                conn.execute(sql.text("""SELECT Vec_Dims("[1]");"""))
            else:
                conn.execute(sql.text("""SELECT 1;"""))
    except exc.DatabaseError as e:
        db_error_code = e.orig.args[0]
        if db_error_code == 1045:
            raise ValueError(
                "Could not connect to the TiDB server. "
                "Please check if the connection string is correct."
            ) from e
        elif db_error_code == 1305:
            raise ValueError(
                "Please confirm if your TiDB supports vector search. "
                "You can check this by running the query `SELECT Vec_Dims('[1]')` in TiDB."
            ) from e
        else:
            raise ValueError(
                "An error occurred while checking the database availability."
            ) from e


def get_or_create(session: Session, model, defaults=None, **kwargs):
    instance = session.query(model).filter_by(**kwargs).first()
    if instance:
        return instance, False
    else:
        if defaults:
            kwargs.update(defaults)
        instance = model(**kwargs)
        session.add(instance)
        session.commit()
        return instance, True
