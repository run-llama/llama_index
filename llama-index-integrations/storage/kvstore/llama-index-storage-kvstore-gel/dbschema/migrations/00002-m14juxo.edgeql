CREATE MIGRATION m14juxosth6svi2fjvhwjmkdaiztzfe2vgj4t2b7vevufcc3fvjdsq
    ONTO m1qleqautnqhkaeegnqlaow5gmykde5etaydswwxa3pvbxlc4jnqjq
{
  ALTER TYPE default::Record {
      ALTER PROPERTY key {
          CREATE CONSTRAINT std::exclusive;
      };
  };
};
