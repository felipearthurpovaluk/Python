import socket
import campo
import time
import os

print('Aguardando cliente se conectar...')
conexao = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

ip = '127.0.0.1'
porta = 50400
origem = (ip,porta)

conexao.bind(origem)
conexao.listen(1)
[sock_dados, info_cliente] = conexao.accept()

jogador = campo.cenario()
adversario = campo.cenario()


#limpa a tela
os.system('cls' if os.name == 'nt' else 'clear')
print('---- BATALHA NAVAL ----')
print('Selecione as posições do seu navio por linha coluna e direção (B=Baixo, D=Direita). Exemplo: C3B')

print('\n---- VOCE -----')
jogador.exibe()

for i in range(jogador.nr_barcos):
    st_sucesso = -1
    while st_sucesso == -1:
        navio = input('Selecione a posição do Navio '+str(i+1)+': ')
        #set_navio(self,posX,posY,tam,dir='rigth'):
        try:
            dir = 'rigth'
            if len(navio) == 3 and navio[2] == 'B':
                dir = 'down'

            st_sucesso = jogador.set_navio(navio[0],navio[1],i+1,dir)

            if st_sucesso == -1:
                print('Posição e ou direção inválida para o navio, insira novamente.')
        except:
            print('Posição e ou direção inválida para o navio, insira novamente.')
        
    os.system('cls' if os.name == 'nt' else 'clear')
    print('---- BATALHA NAVAL ----')
    print('Selecione as posições do seu navio por linha coluna e direção (B=Baixo, D=Direita). Exemplo: C3B')

    print('\n---- VOCE -----')
    jogador.exibe()

print('Aguardando o adversário inserir os navios.')

ok_serv = 1
finalizado_bytes = ok_serv.to_bytes(1,byteorder='big',signed=True)
sock_dados.send(finalizado_bytes)

ok_client = sock_dados.recv(1)

print('Campo de batalha definido. Começando o jogo.')
time.sleep(2)


finalizado = False
ganhou     = False

def atualiza_tela():
    os.system('cls' if os.name == 'nt' else 'clear')
    print('---- VOCE -----')
    jogador.exibe()
    print('\n-- ADVERSARIO --')
    adversario.exibe()
    print('\n1,2,3 = Navio | * = Tiro na agua | @ = Tiro no navio | . = Desconhecido.')


while not finalizado:

    atualiza_tela()
    print('Aguardando tiro do adversário..')

    # atualizando tiro recebido
    st_sucesso = -1
    while st_sucesso == -1:
        tiro_bytes = sock_dados.recv(2)
        tiro = tiro_bytes.decode(encoding='UTF-8')
        #print('TIRO: '+tiro)
        st_sucesso = jogador.retorna_tiro(tiro[0],tiro[1])
        
        #print('st_sucesso (RETORNO) = '+str(st_sucesso))

        retorno_bytes = st_sucesso.to_bytes(2,byteorder='big',signed=True)
        sock_dados.send(retorno_bytes)

        #controle do loop fica só pelo fator -1
        if st_sucesso == -3:
            st_sucesso = -1

        # Caso tenha perdido o jogo
        if st_sucesso == -2:
            ganhou = False
            finalizado = True

    # Finalizando pela condição acima
    if finalizado:
        break

    atualiza_tela()

    # dando um tiro
    st_sucesso = -1
    while st_sucesso == -1:
        try:
            
            tiro = input('Selecione a posição linha coluna para dar um tiro: ')

            px = ord(tiro[0]) - 65 #convertendo de letra para a respectiva linha
            py = int(tiro[1]) - 1

            # enviando dados
            mensagem = tiro.encode(encoding='UTF-8')
            
            sock_dados.send(mensagem)

            tipo_bytes = sock_dados.recv(2)
            st_sucesso = int.from_bytes(tipo_bytes, byteorder='big', signed=True)

            if st_sucesso == -3:
                print('Posição '+tiro+' já foi atirada. Atire novamente.')
                st_sucesso = -1
            elif st_sucesso == -1:
                print('Erro ao atirar na posição '+tiro+'. Atire novamente.')
        except:
            st_sucesso = -1

    px = ord(tiro[0]) - 65 #convertendo de letra para a respectiva linha
    py = int(tiro[1]) - 1

    if st_sucesso == 0:
        adversario.campo[px][py] = '*'
    elif st_sucesso == -2:
        adversario.campo[px][py] = '@'
        finalizado = True
        ganhou = True
    else:
        adversario.campo[px][py] = '@'

atualiza_tela()
if ganhou:
    print('----  Você GANHOU :)  ----')
else:
    print('----  Você PERDEU :/  ----')

input('Pressiona qualquer tecla para sair...')

sock_dados.close()
conexao.close()