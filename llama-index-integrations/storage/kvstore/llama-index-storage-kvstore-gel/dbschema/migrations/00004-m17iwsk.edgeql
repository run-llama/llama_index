CREATE MIGRATION m17iwsk7igiklfqf466lpr4rv5ztskmanpb5tq37y4noatdj24d3tq
    ONTO m1lr5dvz65kkejg7a3pd54ymwj37k3r7crved5cl22zg6tqwxxf7ea
{
  ALTER TYPE default::Record {
      CREATE PROPERTY global_key := (((.namespace ++ '::') ++ .key));
  };
  ALTER TYPE default::Record {
      CREATE CONSTRAINT std::exclusive ON (.global_key);
  };
  ALTER TYPE default::Record {
      DROP CONSTRAINT std::exclusive ON ((.key, .namespace));
  };
};
