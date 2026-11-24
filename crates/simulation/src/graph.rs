pub struct Graph {
    graph: Vec<Vec<bool>>,
}

impl Graph {
    pub fn new(size: usize) -> Self {
        Self {
            graph: vec![vec![true; size]; size],
        }
    }

    pub fn set_visibility(&mut self, craft1: usize, craft2: usize, value: bool) {
        self.graph[craft1][craft2] = value;
        self.graph[craft2][craft1] = value;
    }

    pub fn get_graph(&self) -> &Vec<Vec<bool>> {
        &self.graph
    }
}
