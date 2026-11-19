package ticTacToe;

//import java.util.HashMap;
import java.util.List;
import java.util.Random;

/**
 * A Q-Learning agent with a Q-Table, i.e. a table of Q-Values. This table is implemented in the {@link QTable} class.
 * 
 *  The methods to implement are: 
 * (1) {@link QLearningAgent#train}
 * (2) {@link QLearningAgent#extractPolicy}
 * 
 * Your agent acts in a {@link TTTEnvironment} which provides the method {@link TTTEnvironment#executeMove} which returns an {@link Outcome} object, in other words
 * an [s,a,r,s']: source state, action taken, reward received, and the target state after the opponent has played their move. You may want/need to edit
 * {@link TTTEnvironment} - but you probably won't need to. 
 * @author ae187
 */

public class QLearningAgent extends Agent {
	
	/**
	 * The learning rate, between 0 and 1.
	 */
	double alpha=0.3;
	
	/**
	 * The number of episodes to train for
	 */
	int numEpisodes=10000;
	
	/**
	 * The discount factor (gamma)
	 */
	double discount=0.9;
	
	/**
	 * The epsilon in the epsilon greedy policy used during training.
	 */
	double epsilon=0.6;
	
	/**
	 * This is the Q-Table. To get an value for an (s,a) pair, i.e. a (game, move) pair.
	 */
	QTable qTable=new QTable();
	
	/**
	 * This is the Reinforcement Learning environment that this agent will interact with when it is training.
	 * By default, the opponent is the random agent which should make your q learning agent learn the same policy 
	 * as your value iteration and policy iteration agents.
	 */
	TTTEnvironment env=new TTTEnvironment();
	
	/**
	 * Construct a Q-Learning agent that learns from interactions with {@code opponent}.
	 * @param opponent the opponent agent that this Q-Learning agent will interact with to learn.
	 * @param learningRate This is the rate at which the agent learns. Alpha from your lectures.
	 * @param numEpisodes The number of episodes (games) to train for
	 */
	public QLearningAgent(Agent opponent, double learningRate, int numEpisodes, double discount)
	{
		env=new TTTEnvironment(opponent);
		this.alpha=learningRate;
		this.numEpisodes=numEpisodes;
		this.discount=discount;
		initQTable();
		train();
	}
	
	/**
	 * Initialises all valid q-values -- Q(g,m) -- to 0.
	 *  
	 */
	protected void initQTable()
	{
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
		{
			List<Move> moves=g.getPossibleMoves();
			for(Move m: moves)
			{
				this.qTable.addQValue(g, m, 0.0);
				//System.out.println("initing q value. Game:"+g);
				//System.out.println("Move:"+m);
			}
		}
	}
	
	/**
	 * Uses default parameters for the opponent (a RandomAgent) and the learning rate (0.2). Use other constructor to set these manually.
	 */
	public QLearningAgent()
	{
		this(new RandomAgent(), 0.1, 100, 0.9);
	}
	
	/**
	 *  Implement this method. It should play {@code this.numEpisodes} episodes of Tic-Tac-Toe with the TTTEnvironment, updating q-values according 
	 *  to the Q-Learning algorithm as required. The agent should play according to an epsilon-greedy policy where with the probability {@code epsilon} the
	 *  agent explores, and with probability {@code 1-epsilon}, it exploits. 
	 *  
	 *  At the end of this method you should always call the {@code extractPolicy()} method to extract the policy from the learned q-values. This is currently
	 *  done for you on the last line of the method.
	 */
	public void train()
	{
		this.numEpisodes = 50000; //number of episodes
		Random random = new Random();
		
		for( int i = 0; i < this.numEpisodes; i++ ){ 
			
			Game currGame = this.env.getCurrentGameState(); //current game (state)

			while( !currGame.isTerminal() ){

				List <Move> allMoves = currGame.getPossibleMoves(); 
				Move nextMove = null;
				double rand = random.nextDouble(); 

				//epsilon greedy impelementation

				//exploration : when a randomly generated value is less than epsilon
				if( rand < this.epsilon ){
					if( allMoves.size() != 0 ){
						//generates a random value for getting a random move to be explored
						int index = random.nextInt(allMoves.size());
						nextMove = allMoves.get(index);
					}
				}

				//exploitaion : when a randomly generated value is more than epsilon
				else {
					double maxVal = Double.NEGATIVE_INFINITY;
					//pick a move with the highest Q value to be exploited
					for( Move move : allMoves ){
						double value = this.qTable.getQValue(currGame, move);
						if( value > maxVal ){
							maxVal = value;
							nextMove = move;
						}
					}
				}	
				
				Outcome results = null;
				
				try {
					//gets the outcomes from the executed move
					results = this.env.executeMove(nextMove);
					double maxQ = 0.0;

					if( !results.sPrime.isTerminal() ){

						//evaluates the move with the maximum Q value for the destination state (S prime)
						maxQ = Double.NEGATIVE_INFINITY;
						for( Move movePrime : results.sPrime.getPossibleMoves() ){
							double q = this.qTable.getQValue(results.sPrime, movePrime);
							maxQ = Math.max(maxQ, q);
						}
					}
					double currQ = qTable.getQValue(results.s, results.move); //current state Q value
					//calculating Q values 
					double sample = results.localReward + ( this.discount * maxQ ); 
					double Qvalue = ( 1 - this.alpha ) * currQ + this.alpha * sample;

					//updating Q values : adds the new results to the Q table
					this.qTable.addQValue(results.s, results.move, Qvalue);

				} catch( IllegalMoveException e ){
					e.printStackTrace();
				}					
			}
			//resets environment for a new episode
			this.env.reset();
		}
		
		this.policy = extractPolicy();
		if( this.policy == null ){
			System.out.println("Unimplemented methods! First implement the train() & extractPolicy methods");
		}
	}
	
	/** Implement this method. It should use the q-values in the {@code qTable} to extract a policy and return it.
	 *
	 * @return the policy currently inherent in the QTable
	 */
	public Policy extractPolicy()
	{
		Policy newPolicy = new Policy();

		//goes through all the games (states) from the Q table
		for( Game game : this.qTable.keySet() ){

			//skips if the game is a terminal state
			if( game.isTerminal() ){ continue; }

			Move bestMove = null;
			double maxVal = Double.NEGATIVE_INFINITY;
			
			//goes through all possible moves for that state
			for( Move move : game.getPossibleMoves() ){

				//gets the move with the maximum Q value 
				double qValue = this.qTable.getQValue(game, move);
				if( qValue > maxVal ){
					maxVal = qValue;
					bestMove = move;
				}
			}
			//updating policy : adds the best move to the policy object
			if( bestMove != null ){
				newPolicy.policy.put(game, bestMove);
			}
		}
		return newPolicy;
	}
	
	public static void main(String a[]) throws IllegalMoveException
	{
		//Test method to play your agent against a human agent (yourself).
		QLearningAgent agent=new QLearningAgent();
		
		HumanAgent d=new HumanAgent();
		
		Game g=new Game(agent, d, d);
		g.playOut();	

		//for testing the Q learning agent against other agents for 50 times and printing the results in the terminal
		QLearningAgent ql = new QLearningAgent();
		
		Agent aa = new RandomAgent();
		int rw = 0;
		int rd = 0;
		int rl = 0;

		for( int i = 0; i < 50; i++ ) {
			Game gg = new Game(ql, aa, ql);
			gg.playOut();
			if( gg.state == Game.X_WON ) {
				rw++;
			} else if( gg.state == Game.O_WON ) {
				rl++;
			} else { rd++; }
		}
		
		aa = new AggressiveAgent();
		int aw = 0;
		int ad = 0;
		int al = 0;
		for( int i = 0; i < 50; i++ ){
			Game gg = new Game(ql, aa, ql);
			gg.playOut();
			if (gg.state == Game.X_WON) {
				aw++;
			} else if (gg.state == Game.O_WON) {
				al++;
			} else { ad++; }
		}
		
		aa = new DefensiveAgent();
		int dw = 0;
		int dd = 0;
		int dl = 0;
		for( int i = 0; i < 50; i++ ){
			Game gg = new Game(ql, aa, ql);
			gg.playOut();
			if (gg.state == Game.X_WON) {
				dw++;
			} else if (gg.state == Game.O_WON) {
				dl++;
			} else {
				dd++;
			}
		}

		System.err.println("Q learning agent playing against:");
		System.err.println("");
		System.err.println("-------------------------------");
		System.err.println("Against Random agent:");
		System.err.println("Wins: " + rw + ", Losses: " + rl + ", Draws: " + rd);
		System.err.println("");

		System.err.println("-------------------------------");
		System.err.println("Against Aggresive agent:");
		System.err.println("Wins: " + aw + ", Losses: " + al + ", Draws: " + ad);
		System.err.println("");

		System.err.println("-------------------------------");
		System.err.println("Against Defensive agent:");
		System.err.println("Wins: " + dw + ", Losses: " + dl + ", Draws: " + dd);
		System.err.println("");
		System.err.println("-------------------------------");

	}
}
