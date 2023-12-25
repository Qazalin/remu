
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_4n115>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mov_b32 s2, s15                                          // 000000001714: BE82000F
	s_ashr_i32 s3, s15, 31                                     // 000000001718: 86039F0F
	v_mov_b32_e32 v0, 0                                        // 00000000171C: 7E000280
	s_lshl_b64 s[8:9], s[2:3], 1                               // 000000001720: 84888102
	s_waitcnt lgkmcnt(0)                                       // 000000001724: BF89FC07
	s_add_u32 s6, s6, s8                                       // 000000001728: 80060806
	s_addc_u32 s7, s7, s9                                      // 00000000172C: 82070907
	s_add_u32 s0, s0, s8                                       // 000000001730: 80000800
	s_addc_u32 s1, s1, s9                                      // 000000001734: 82010901
	s_clause 0x1                                               // 000000001738: BF850001
	global_load_u16 v1, v0, s[6:7]                             // 00000000173C: DC4A0000 01060000
	global_load_i16 v2, v0, s[0:1]                             // 000000001744: DC4E0000 02000000
	s_lshl_b64 s[0:1], s[2:3], 2                               // 00000000174C: 84808202
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001750: BF870009
	s_add_u32 s0, s4, s0                                       // 000000001754: 80000004
	s_addc_u32 s1, s5, s1                                      // 000000001758: 82010105
	s_waitcnt vmcnt(0)                                         // 00000000175C: BF8903F7
	v_mul_lo_u32 v1, v2, v1                                    // 000000001760: D72C0001 00020302
	global_store_b32 v0, v1, s[0:1]                            // 000000001768: DC6A0000 00000100
	s_nop 0                                                    // 000000001770: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001774: BFB60003
	s_endpgm                                                   // 000000001778: BFB00000
