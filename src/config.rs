use std::fs;

use toml;


pub fn from_file<T>(file : &str) -> Result<T, toml::de::Error>
where 
    T : serde::de::DeserializeOwned
{
    let toml_str = fs::read_to_string(file)
    .expect("FileException: Something occur file...");

    toml::from_str(&toml_str)
}

#[cfg(test)]
mod configuration {

    use serde::Deserialize;

    use super::from_file;

    #[derive(Deserialize)]
    struct TestConfig {
        _name_or_path : String
    } 
    #[test]
    fn build_default(){
        let config : TestConfig = from_file("src/models/phi-2/config.toml").unwrap();
        assert_eq!(config._name_or_path, "microsoft/phi-2")
    }



}