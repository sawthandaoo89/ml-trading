""""""  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""Assess a betting strategy.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		 	 	 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		 	 	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 		  		  		    	 		 		   		 		  
or edited.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		 	 	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Student Name: Saw Thanda Oo		  	   		 	 	 		  		  		    	 		 		   		 		  
GT User ID: soo7		  	   		 	 	 		  		  		    	 		 		   		 		  
GT ID: 904056931		  	   		 	 	 		  		  		    	 		 		   		 		  
"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import numpy as np  
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
 
def author():  
    """  
    :return: The GT username of the student  
    :rtype: str  
    """  
    return "soo7" 
 
 
# Optional study group API per assignment spec
def study_group():
    """
    :return: List of GT usernames of study group members
    :rtype: list[str]
    """
    return []


def gtid():  
    """  
    :return: The GT ID of the student  
    :rtype: int  
    """  
    return 904056931 
 
 
def get_spin_result(win_prob):  
    """  
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  
 
    :param win_prob: The probability of winning  
    :type win_prob: float  
    :return: The result of the spin.  
    :rtype: bool  
    """  
    result = False  
    if np.random.random() <= win_prob:  
        result = True  
    return result  
 
 
def simulate_episode_unlimited(max_spins: int = 1000, target_win: int = 80, win_prob: float = 18.0 / 38.0) -> np.ndarray:
    """Simulate a single episode (unlimited bankroll) with Martingale strategy.

    Returns an array of length max_spins+1 with winnings[0] == 0 and
    winnings[i] the cumulative winnings after i-th spin; fill-forward after stopping.
    """
    winnings = np.zeros(max_spins + 1, dtype=int)
    episode_winnings = 0
    spins = 0

    while spins < max_spins and episode_winnings < target_win:
        won = False
        bet_amount = 1
        while (not won) and spins < max_spins:
            bet = bet_amount
            won = get_spin_result(win_prob)
            if won:
                episode_winnings += bet
            else:
                episode_winnings -= bet
                bet_amount *= 2
            spins += 1
            winnings[spins] = episode_winnings

    if spins < max_spins:
        winnings[spins + 1 :] = episode_winnings

    return winnings
 
 
def simulate_episode_limited(max_spins: int = 1000, target_win: int = 80, bankroll: int = 256, win_prob: float = 18.0 / 38.0) -> np.ndarray:
    """Simulate a single episode (limited bankroll) with Martingale strategy.

    Bankroll starts at `bankroll`; track winnings relative to 0; stop when winnings >= target
    or bankroll depleted (winnings == -bankroll). Fill-forward after stopping.
    """
    winnings = np.zeros(max_spins + 1, dtype=int)
    episode_winnings = 0
    spins = 0

    def remaining_bankroll() -> int:
        return bankroll + episode_winnings

    while spins < max_spins and episode_winnings < target_win and remaining_bankroll() > 0:
        won = False
        bet_amount = 1
        while (not won) and spins < max_spins and remaining_bankroll() > 0:
            bet = min(bet_amount, remaining_bankroll())
            if bet <= 0:
                break
            won = get_spin_result(win_prob)
            if won:
                episode_winnings += bet
            else:
                episode_winnings -= bet
                bet_amount *= 2
            spins += 1
            winnings[spins] = episode_winnings

    if spins < max_spins:
        # Ensure lower bound at -bankroll
        episode_winnings = max(episode_winnings, -bankroll)
        winnings[spins + 1 :] = episode_winnings

    return winnings
 
 
def plot_episodes(episodes: np.ndarray, title: str, filename: str, xlabel: str = "Spin", ylabel: str = "Winnings ($)", xlim: tuple = (0, 300), ylim: tuple = (-256, 100)) -> None:
    os.makedirs(os.path.join(os.path.dirname(__file__), "images"), exist_ok=True)
    plt.figure(figsize=(8, 5))
    num_eps = episodes.shape[0]
    x = np.arange(episodes.shape[1])
    for i in range(num_eps):
        plt.plot(x, episodes[i, :], label=f"Episode {i+1}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.legend(ncol=2, fontsize="small")
    out_path = os.path.join(os.path.dirname(__file__), "images", filename)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
 
 
def plot_stats(curves: np.ndarray, center: str, title: str, filename: str, xlabel: str = "Spin", ylabel: str = "Winnings ($)", xlim: tuple = (0, 300), ylim: tuple = (-256, 100)) -> None:
    """Plot mean/median and +/- std bands across episodes for each spin index."""
    os.makedirs(os.path.join(os.path.dirname(__file__), "images"), exist_ok=True)
    center_func = np.mean if center == "mean" else np.median
    center_vals = center_func(curves, axis=0)
    std_vals = np.std(curves, axis=0, ddof=0)
    x = np.arange(curves.shape[1])
    plt.figure(figsize=(8, 5))
    plt.plot(x, center_vals, label=center.capitalize())
    plt.plot(x, center_vals + std_vals, label=f"{center.capitalize()} + Std")
    plt.plot(x, center_vals - std_vals, label=f"{center.capitalize()} - Std")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.legend()
    out_path = os.path.join(os.path.dirname(__file__), "images", filename)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
 
 
def test_code():  
    """  
    Method to test your code  
    """  
    # Set seed ONCE for reproducibility
    np.random.seed(gtid())

    win_prob = 18.0 / 38.0  # American roulette probability of black

    # Experiment 1
    # Figure 1: 10 episodes, plot all
    eps10 = np.vstack([simulate_episode_unlimited(win_prob=win_prob) for _ in range(10)])
    plot_episodes(eps10, title="Figure 1: 10 Episodes (Unlimited Bankroll)", filename="figure1.png")

    # Figure 2 and 3: 1000 episodes unlimited
    eps1000_unlim = np.vstack([simulate_episode_unlimited(win_prob=win_prob) for _ in range(1000)])
    plot_stats(eps1000_unlim, center="mean", title="Figure 2: Mean ± Std (Unlimited Bankroll)", filename="figure2.png")
    plot_stats(eps1000_unlim, center="median", title="Figure 3: Median ± Std (Unlimited Bankroll)", filename="figure3.png")

    # Experiment 2 (realistic, bankroll = 256)
    eps1000_lim = np.vstack([simulate_episode_limited(win_prob=win_prob) for _ in range(1000)])
    plot_stats(eps1000_lim, center="mean", title="Figure 4: Mean ± Std (Bankroll 256)", filename="figure4.png")
    plot_stats(eps1000_lim, center="median", title="Figure 5: Median ± Std (Bankroll 256)", filename="figure5.png")

    # Compute and save key metrics
    exp1_prob_exact80 = float(np.mean(eps1000_unlim[:, -1] == 80))
    exp1_prob_ge80 = float(np.mean(eps1000_unlim[:, -1] >= 80))
    exp1_expected = float(np.mean(eps1000_unlim[:, -1]))

    exp2_prob_exact80 = float(np.mean(eps1000_lim[:, -1] == 80))
    exp2_prob_ge80 = float(np.mean(eps1000_lim[:, -1] >= 80))
    exp2_expected = float(np.mean(eps1000_lim[:, -1]))

    results_path = os.path.join(os.path.dirname(__file__), "p1_results.txt")
    with open(results_path, "w") as f:
        f.write("Experiment 1 (Unlimited Bankroll)\n")
        f.write(f"P(final == 80): {exp1_prob_exact80:.6f}\n")
        f.write(f"P(final >= 80): {exp1_prob_ge80:.6f}\n")
        f.write(f"E[final winnings]: {exp1_expected:.6f}\n\n")
        f.write("Experiment 2 (Bankroll = 256)\n")
        f.write(f"P(final == 80): {exp2_prob_exact80:.6f}\n")
        f.write(f"P(final >= 80): {exp2_prob_ge80:.6f}\n")
        f.write(f"E[final winnings]: {exp2_expected:.6f}\n")
 
 
if __name__ == "__main__":  
    test_code()  
