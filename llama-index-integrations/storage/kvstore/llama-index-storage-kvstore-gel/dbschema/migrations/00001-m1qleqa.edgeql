CREATE MIGRATION m1qleqautnqhkaeegnqlaow5gmykde5etaydswwxa3pvbxlc4jnqjq
    ONTO initial
{
  CREATE FUTURE simple_scoping;
  CREATE TYPE default::Record {
      CREATE REQUIRED PROPERTY key: std::str;
      CREATE REQUIRED PROPERTY namespace: std::str;
      CREATE PROPERTY value: std::json;
  };
};
