CREATE MIGRATION m1lr5dvz65kkejg7a3pd54ymwj37k3r7crved5cl22zg6tqwxxf7ea
    ONTO m14juxosth6svi2fjvhwjmkdaiztzfe2vgj4t2b7vevufcc3fvjdsq
{
  ALTER TYPE default::Record {
      CREATE CONSTRAINT std::exclusive ON ((.key, .namespace));
      ALTER PROPERTY key {
          DROP CONSTRAINT std::exclusive;
      };
  };
};
