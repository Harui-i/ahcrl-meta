use ahc061_rl_tools::vec_env::EnvSlot;
use tools::{compute_score_details, parse_output};

#[test]
fn env_score_matches_official_parser_for_fixed_actions() {
    let mut env = EnvSlot::from_seed(0, Some(4), Some(3));
    for _ in 0..env.input.T {
        let mask = env.mask();
        let action = mask
            .iter()
            .position(|&ok| ok)
            .expect("player0 must have at least one legal action");
        env.step_action_index(action).unwrap();
    }

    let official_out = parse_output(&env.input, &env.output_text()).unwrap();
    let (official_score, err, _scores) = compute_score_details(&env.input, &official_out.out);
    assert_eq!(err, "");
    assert_eq!(env.score(), official_score);
}
