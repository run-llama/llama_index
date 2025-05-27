CREATE MIGRATION m1vdszejnp3krm4e7seblzye5kwcosus24suik3iirkbezg6drfepq
    ONTO m17iwsk7igiklfqf466lpr4rv5ztskmanpb5tq37y4noatdj24d3tq
{
  ALTER TYPE default::Record {
      DROP CONSTRAINT std::exclusive ON (.global_key);
  };
  ALTER TYPE default::Record {
      CREATE CONSTRAINT std::exclusive ON ((.key, .namespace));
      DROP PROPERTY global_key;
  };
};
