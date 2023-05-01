using Documenter, SupplyChains

makedocs(
    sitename="Supply Chains",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    )
)
