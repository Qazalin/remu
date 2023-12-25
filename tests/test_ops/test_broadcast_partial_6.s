
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_4_5n1>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mov_b32 s2, s15                                          // 000000001714: BE82000F
	s_ashr_i32 s3, s15, 31                                     // 000000001718: 86039F0F
	s_mul_i32 s10, s15, 5                                      // 00000000171C: 960A850F
	s_lshl_b64 s[8:9], s[2:3], 2                               // 000000001720: 84888202
	v_mov_b32_e32 v1, 0                                        // 000000001724: 7E020280
	s_waitcnt lgkmcnt(0)                                       // 000000001728: BF89FC07
	s_add_u32 s2, s6, s8                                       // 00000000172C: 80020806
	s_addc_u32 s3, s7, s9                                      // 000000001730: 82030907
	s_add_i32 s6, s10, s14                                     // 000000001734: 81060E0A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001738: BF870499
	s_ashr_i32 s7, s6, 31                                      // 00000000173C: 86079F06
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001740: 84868206
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001744: BF870009
	s_add_u32 s0, s0, s6                                       // 000000001748: 80000600
	s_addc_u32 s1, s1, s7                                      // 00000000174C: 82010701
	s_load_b32 s2, s[2:3], null                                // 000000001750: F4000081 F8000000
	s_load_b32 s0, s[0:1], null                                // 000000001758: F4000000 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000001760: BF89FC07
	v_sub_f32_e64 v0, s2, s0                                   // 000000001764: D5040000 00000002
	s_add_u32 s0, s4, s6                                       // 00000000176C: 80000604
	s_addc_u32 s1, s5, s7                                      // 000000001770: 82010705
	global_store_b32 v1, v0, s[0:1]                            // 000000001774: DC6A0000 00000001
	s_nop 0                                                    // 00000000177C: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001780: BFB60003
	s_endpgm                                                   // 000000001784: BFB00000
