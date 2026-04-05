use proc_macro::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream, Result as SynResult};
use syn::parse_macro_input;
use syn::token::Comma;
use syn::{AttributeArgs, ItemFn, Lit, Meta, NestedMeta};

struct ChatMacroArgs {
    msg: String,
    model_type: String,
}

impl Parse for ChatMacroArgs {
    fn parse(input: ParseStream) -> SynResult<Self> {
        let mut msg = String::new();
        let mut model_type = String::new();

        while !input.is_empty() {
            let ident: syn::Ident = input.parse()?;
            input.parse::<syn::Token![=]>()?;
            let value: syn::LitStr = input.parse()?;

            match ident.to_string().as_str() {
                "msg" => msg = value.value(),
                "type" => model_type = value.value(),
                _ => (),
            }

            if input.peek(Comma) {
                input.parse::<Comma>()?;
            }
        }

        Ok(ChatMacroArgs { msg, model_type })
    }
}

#[proc_macro_attribute]
pub fn chat(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as ChatMacroArgs);
    let input_fn = parse_macro_input!(item as ItemFn);

    let msg_text = args.msg;
    let model_type = args.model_type;

    let fn_name = &input_fn.sig.ident;
    let vis = &input_fn.vis;
    let inputs = &input_fn.sig.inputs;
    let output = &input_fn.sig.output;
    let block = &input_fn.block;
    let attrs = &input_fn.attrs;

    let model_creation = match model_type.as_str() {
        "openai" => quote! {
            let api_key = std::env::var("OPENAI_API_KEY")
                .expect("OPENAI_API_KEY environment variable not set");
            let llm = ::langhub::core::llm::OpenAI::new(api_key)
                .gpt4_turbo()
                .with_temperature(0.7);
        },
        "deepseek" => quote! {
            let api_key = std::env::var("DEEPSEEK_API_KEY")
                .expect("DEEPSEEK_API_KEY environment variable not set");
            let llm = ::langhub::core::llm::DeepSeek::new(api_key)
                .chat_model()
                .with_temperature(0.7);
        },
        "anthropic" => quote! {
            let api_key = std::env::var("ANTHROPIC_API_KEY")
                .expect("ANTHROPIC_API_KEY environment variable not set");
            let llm = ::langhub::core::llm::Anthropic::new(api_key)
                .claude3_sonnet()
                .with_temperature(0.7);
        },
        "google" => quote! {
            let api_key = std::env::var("GOOGLE_API_KEY")
                .expect("GOOGLE_API_KEY environment variable not set");
            let llm = ::langhub::core::llm::GoogleAI::new(api_key)
                .gemini15_pro()
                .with_temperature(0.7);
        },
        "cohere" => quote! {
            let api_key = std::env::var("COHERE_API_KEY")
                .expect("COHERE_API_KEY environment variable not set");
            let llm = ::langhub::core::llm::Cohere::new(api_key)
                .command()
                .with_temperature(0.7);
        },
        "huggingface" => quote! {
            let api_key = std::env::var("HUGGINGFACE_API_KEY")
                .expect("HUGGINGFACE_API_KEY environment variable not set");
            let llm = ::langhub::core::llm::HuggingFace::new(api_key)
                .llama3_8b()
                .with_temperature(0.7);
        },
        "azure" => quote! {
            let api_key = std::env::var("AZURE_API_KEY")
                .expect("AZURE_API_KEY environment variable not set");
            let endpoint = std::env::var("AZURE_ENDPOINT")
                .expect("AZURE_ENDPOINT environment variable not set");
            let deployment = std::env::var("AZURE_DEPLOYMENT_NAME")
                .expect("AZURE_DEPLOYMENT_NAME environment variable not set");
            let llm = ::langhub::core::llm::AzureOpenAI::new(api_key, endpoint, deployment);
        },
        "mistral" => quote! {
            let api_key = std::env::var("MISTRAL_API_KEY")
                .expect("MISTRAL_API_KEY environment variable not set");
            let llm = ::langhub::core::llm::Mistral::new(api_key)
                .small()
                .with_temperature(0.7);
        },
        "groq" => quote! {
            let api_key = std::env::var("GROQ_API_KEY")
                .expect("GROQ_API_KEY environment variable not set");
            let llm = ::langhub::core::llm::Groq::new(api_key)
                .mixtral()
                .with_temperature(0.7);
        },
        "together" => quote! {
            let api_key = std::env::var("TOGETHER_API_KEY")
                .expect("TOGETHER_API_KEY environment variable not set");
            let llm = ::langhub::core::llm::Together::new(api_key)
                .mixtral()
                .with_temperature(0.7);
        },
        "replicate" => quote! {
            let api_key = std::env::var("REPLICATE_API_KEY")
                .expect("REPLICATE_API_KEY environment variable not set");
            let llm = ::langhub::core::llm::Replicate::new(api_key)
                .mixtral()
                .with_temperature(0.7);
        },
        "fireworks" => quote! {
            let api_key = std::env::var("FIREWORKS_API_KEY")
                .expect("FIREWORKS_API_KEY environment variable not set");
            let llm = ::langhub::core::llm::Fireworks::new(api_key)
                .mixtral()
                .with_temperature(0.7);
        },
        "perplexity" => quote! {
            let api_key = std::env::var("PERPLEXITY_API_KEY")
                .expect("PERPLEXITY_API_KEY environment variable not set");
            let llm = ::langhub::core::llm::Perplexity::new(api_key)
                .sonar_medium()
                .with_temperature(0.7);
        },
        "baidu" => quote! {
            let api_key = std::env::var("BAIDU_API_KEY")
                .expect("BAIDU_API_KEY environment variable not set");
            let secret_key = std::env::var("BAIDU_SECRET_KEY")
                .expect("BAIDU_SECRET_KEY environment variable not set");
            let llm = ::langhub::core::llm::BaiduWenxin::new(api_key, secret_key)
                .ernie4_0();
        },
        "alibaba" => quote! {
            let api_key = std::env::var("ALIBABA_API_KEY")
                .expect("ALIBABA_API_KEY environment variable not set");
            let llm = ::langhub::core::llm::AlibabaTongyi::new(api_key)
                .qwen_plus()
                .with_temperature(0.7);
        },
        "tencent" => quote! {
            let secret_id = std::env::var("TENCENT_SECRET_ID")
                .expect("TENCENT_SECRET_ID environment variable not set");
            let secret_key = std::env::var("TENCENT_SECRET_KEY")
                .expect("TENCENT_SECRET_KEY environment variable not set");
            let llm = ::langhub::core::llm::TencentHunyuan::new(secret_id, secret_key)
                .hunyuan_pro();
        },
        "zhipu" => quote! {
            let api_key = std::env::var("ZHIPU_API_KEY")
                .expect("ZHIPU_API_KEY environment variable not set");
            let llm = ::langhub::core::llm::ZhipuAI::new(api_key)
                .glm4()
                .with_temperature(0.7);
        },
        "minimax" => quote! {
            let api_key = std::env::var("MINIMAX_API_KEY")
                .expect("MINIMAX_API_KEY environment variable not set");
            let group_id = std::env::var("MINIMAX_GROUP_ID")
                .expect("MINIMAX_GROUP_ID environment variable not set");
            let llm = ::langhub::core::llm::MiniMax::new(api_key, group_id)
                .abab6_5();
        },
        "moonshot" => quote! {
            let api_key = std::env::var("MOONSHOT_API_KEY")
                .expect("MOONSHOT_API_KEY environment variable not set");
            let llm = ::langhub::core::llm::Moonshot::new(api_key)
                .kimi_128k()
                .with_temperature(0.7);
        },
        "baichuan" => quote! {
            let api_key = std::env::var("BAICHUAN_API_KEY")
                .expect("BAICHUAN_API_KEY environment variable not set");
            let llm = ::langhub::core::llm::Baichuan::new(api_key)
                .baichuan4()
                .with_temperature(0.7);
        },
        "yi" => quote! {
            let api_key = std::env::var("YI_API_KEY")
                .expect("YI_API_KEY environment variable not set");
            let llm = ::langhub::core::llm::Yi::new(api_key)
                .yi34b()
                .with_temperature(0.7);
        },
        _ => quote! {
            let api_key = std::env::var("OPENAI_API_KEY")
                .expect("OPENAI_API_KEY environment variable not set");
            let llm = ::langhub::core::llm::OpenAI::new(api_key)
                .gpt4_turbo()
                .with_temperature(0.7);
        },
    };

    let expanded = quote! {
        #(#attrs)*
        #vis fn #fn_name(#inputs) #output {
            use ::langhub::core::llm::LLM;

            #model_creation

            let result = ::tokio::runtime::Runtime::new()
                .expect("Failed to create Tokio runtime")
                .block_on(async move {
                    let response = llm.generate(#msg_text).await?;
                    ::langhub::types::Result::Ok(response.text)
                });

            let llm_response = match result {
                ::langhub::types::Result::Ok(text) => text,
                ::langhub::types::Result::Err(e) => {
                    eprintln!("LLM error: {}", e);
                    return Err(e.into());
                }
            };

            #block
        }
    };

    expanded.into()
}
