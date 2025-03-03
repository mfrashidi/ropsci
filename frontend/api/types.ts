export interface IPredictionRequest {
  image: string;
  name: string;
  round: string;
  game: string;
}

export interface IPredictionResponse {
  rock: number;
  paper: number;
  scissors: number;
}

export interface IPredictionRequestResponse {
  data: IPredictionResponse;
}