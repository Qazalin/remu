
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_4_4n17>:
	s_load_b128 s[4:7], s[0:1], null                           // 000000001700: F4080100 F8000000
	s_mov_b32 s2, s15                                          // 000000001708: BE82000F
	s_ashr_i32 s15, s14, 31                                    // 00000000170C: 860F9F0E
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001710: BF870009
	s_lshl_b64 s[0:1], s[14:15], 3                             // 000000001714: 8480830E
	s_waitcnt lgkmcnt(0)                                       // 000000001718: BF89FC07
	s_add_u32 s6, s6, s0                                       // 00000000171C: 80060006
	s_addc_u32 s7, s7, s1                                      // 000000001720: 82070107
	s_load_b64 s[6:7], s[6:7], null                            // 000000001724: F4040183 F8000000
	s_waitcnt lgkmcnt(0)                                       // 00000000172C: BF89FC07
	s_add_u32 s6, s6, 1                                        // 000000001730: 80068106
	s_addc_u32 s7, s7, 0                                       // 000000001734: 82078007
	s_lshl_b32 s2, s2, 2                                       // 000000001738: 84028202
	v_mov_b32_e32 v0, s6                                       // 00000000173C: 7E000206
	s_ashr_i32 s3, s2, 31                                      // 000000001740: 86039F02
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v1, s7              // 000000001744: CA100080 02000007
	s_lshl_b64 s[2:3], s[2:3], 3                               // 00000000174C: 84828302
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001750: BF870009
	s_add_u32 s2, s4, s2                                       // 000000001754: 80020204
	s_addc_u32 s3, s5, s3                                      // 000000001758: 82030305
	s_add_u32 s0, s2, s0                                       // 00000000175C: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000001760: 82010103
	global_store_b64 v2, v[0:1], s[0:1]                        // 000000001764: DC6E0000 00000002
	s_nop 0                                                    // 00000000176C: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001770: BFB60003
	s_endpgm                                                   // 000000001774: BFB00000
