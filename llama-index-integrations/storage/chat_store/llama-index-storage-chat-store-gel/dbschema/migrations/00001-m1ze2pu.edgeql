CREATE MIGRATION m1ze2purgmt2fkajxuidp4bcozwldfc66wr3qw2eo3cfm2pt3o442a
    ONTO initial
{
  CREATE FUTURE simple_scoping;
  CREATE TYPE default::Record {
      CREATE REQUIRED PROPERTY key: std::str {
          CREATE CONSTRAINT std::exclusive;
      };
      CREATE PROPERTY value: array<std::json>;
  };
};
