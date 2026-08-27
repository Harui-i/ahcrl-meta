use super::State;

/// Read-only state needed by an external reinforcement-learning environment.
pub struct StateView<'a> {
    pub food: &'a [Vec<usize>],
    pub positions: &'a [(usize, usize)],
    pub colors: &'a [usize],
    pub turn: usize,
}

pub fn state_view(state: &State) -> StateView<'_> {
    StateView {
        food: &state.f,
        positions: &state.ij,
        colors: &state.c,
        turn: state.turn,
    }
}
