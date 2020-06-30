from utils.compute_gradient_penalty import compute_gradient_penalty

def get_Cη_loss(Cη, η, η_approximation):
    real_η_validity = Cη(η)
    η_approximation_validity = Cη(η_approximation)

    gradient_penalty_η = compute_gradient_penalty(Cη, η.data, η_approximation.data)
    loss_Cη = - real_η_validity.mean() + η_approximation_validity.mean() + 10 * gradient_penalty_η

    return loss_Cη

def get_Cyδ_loss(Cyδ, yδ, y_renoised):
    real_yδ_validity = Cyδ(yδ)
    y_renoised_validity = Cyδ(y_renoised)

    gradient_penalty_yδ = compute_gradient_penalty(Cyδ, yδ.data, y_renoised.data)
    loss_Cyδ = - real_yδ_validity.mean() + y_renoised_validity.mean() + 10 * gradient_penalty_yδ

    return loss_Cyδ

def get_G_loss(G, Cη, Cyδ, yδ):
    _, η_approximation, y_renoised = G.apply_for_training(yδ)
    loss_G = - Cyδ(y_renoised).mean() - Cη(η_approximation).mean()

    return loss_G