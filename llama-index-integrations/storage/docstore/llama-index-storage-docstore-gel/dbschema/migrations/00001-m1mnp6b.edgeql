CREATE MIGRATION m1mnp6bguuljsz2j7iv3xkws43zfiadbqohcg3zzerxtbxivnccaxa
    ONTO initial
{
  CREATE FUTURE simple_scoping;
  CREATE TYPE default::Record {
      CREATE REQUIRED PROPERTY key: std::str;
      CREATE REQUIRED PROPERTY namespace: std::str;
      CREATE CONSTRAINT std::exclusive ON ((.key, .namespace));
      CREATE PROPERTY value: std::json;
  };
};
