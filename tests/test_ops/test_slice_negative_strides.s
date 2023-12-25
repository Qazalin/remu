
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_10_100>:
	s_load_b128 s[4:7], s[0:1], null                           // 000000001700: F4080100 F8000000
	s_mul_i32 s0, s15, 0xffffff9c                              // 000000001708: 9600FF0F FFFFFF9C
	s_mov_b32 s2, s15                                          // 000000001710: BE82000F
	s_ashr_i32 s1, s0, 31                                      // 000000001714: 86019F00
	s_mulk_i32 s2, 0x64                                        // 000000001718: B8020064
	s_lshl_b64 s[0:1], s[0:1], 2                               // 00000000171C: 84808200
	s_waitcnt lgkmcnt(0)                                       // 000000001720: BF89FC07
	s_add_u32 s3, s6, s0                                       // 000000001724: 80030006
	s_addc_u32 s7, s7, s1                                      // 000000001728: 82070107
	s_ashr_i32 s15, s14, 31                                    // 00000000172C: 860F9F0E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001730: BF870499
	s_lshl_b64 s[0:1], s[14:15], 2                             // 000000001734: 8480820E
	s_add_u32 s6, s3, s0                                       // 000000001738: 80060003
	s_addc_u32 s7, s7, s1                                      // 00000000173C: 82070107
	s_ashr_i32 s3, s2, 31                                      // 000000001740: 86039F02
	s_load_b32 s6, s[6:7], 0xe10                               // 000000001744: F4000183 F8000E10
	s_lshl_b64 s[2:3], s[2:3], 2                               // 00000000174C: 84828202
	v_mov_b32_e32 v0, 0                                        // 000000001750: 7E000280
	s_add_u32 s2, s4, s2                                       // 000000001754: 80020204
	s_addc_u32 s3, s5, s3                                      // 000000001758: 82030305
	s_add_u32 s0, s2, s0                                       // 00000000175C: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000001760: 82010103
	s_waitcnt lgkmcnt(0)                                       // 000000001764: BF89FC07
	v_mov_b32_e32 v1, s6                                       // 000000001768: 7E020206
	global_store_b32 v0, v1, s[0:1]                            // 00000000176C: DC6A0000 00000100
	s_nop 0                                                    // 000000001774: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001778: BFB60003
	s_endpgm                                                   // 00000000177C: BFB00000
