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
        self.graph[craft1 as usize][craft2 as usize] = value;
        self.graph[craft2 as usize][craft1 as usize] = value;
    }

    pub fn get_graph(&self) -> &Vec<Vec<bool>> {
        &self.graph
    }
}
