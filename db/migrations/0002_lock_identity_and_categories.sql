CREATE TRIGGER models_append_only
BEFORE UPDATE OR DELETE ON lollanet.models
FOR EACH ROW EXECUTE FUNCTION lollanet.reject_mutation();

CREATE TRIGGER categories_append_only
BEFORE UPDATE OR DELETE ON lollanet.categories
FOR EACH ROW EXECUTE FUNCTION lollanet.reject_mutation();
